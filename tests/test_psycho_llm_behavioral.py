from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from local_inference.prompt_encoding import render_chat_text
from psycho_llm_behavioral.judge_client import JudgeClient, JudgeConfig
from psycho_llm_behavioral.judge_prompt import (
    FACTOR_ORDER,
    parse_judge_response,
    reverse_score,
    sample_keying,
)
from psycho_llm_behavioral.prompts import build_messages, load_behavioral_prompts
from psycho_llm_behavioral.run import build_parser as build_behavioral_parser
from psycho_llm_behavioral.runner import (
    GenerationConfig,
    response_id,
    run_generation,
    run_judging,
)
from psycho_llm_behavioral.storage import JsonlStore, export_results
from value_action_gap_bench.run import build_parser as build_value_action_parser


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def generate_batch(
        self,
        prompts,
        *,
        max_new_tokens,
        temperature,
        use_chat_template,
    ):
        self.calls += 1
        return [f"raw model response {self.calls}:{index}" for index, _ in enumerate(prompts)]


class FakeTokenizer:
    chat_template = "available"

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        return "rendered"


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class FakeOpener:
    def __init__(self):
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((json.loads(request.data), timeout))
        return FakeHTTPResponse(
            {
                "id": "judge-test",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"RE": 1, "DE": 2, "BO": 3, "GU": 4, "VB": 5}'
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )


class PromptTests(unittest.TestCase):
    def test_prompt_pool_matches_paper_design(self):
        prompts = load_behavioral_prompts()
        self.assertEqual(len(prompts), 20)
        self.assertEqual(
            Counter(prompt["dimension_code"] for prompt in prompts),
            {"RE": 4, "DE": 4, "BO": 4, "GU": 4, "VB": 4},
        )
        self.assertEqual(sum(prompt["is_two_turn"] for prompt in prompts), 2)

    def test_two_turn_messages_are_preserved(self):
        prompt = next(
            prompt for prompt in load_behavioral_prompts() if prompt["prompt_id"] == "RE-BP01"
        )
        messages = build_messages(prompt)
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user"])

        tokenizer = FakeTokenizer()
        self.assertEqual(render_chat_text(tokenizer, messages, True), "rendered")
        self.assertEqual(tokenizer.messages, messages)

    def test_unknown_prompt_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown behavioral prompt"):
            load_behavioral_prompts(prompt_ids=["NOPE"])


class JudgePromptTests(unittest.TestCase):
    def test_keying_is_stable_and_reverse_scored(self):
        keying = sample_keying("deepseek-v4-pro", "response-1")
        self.assertEqual(keying, sample_keying("deepseek-v4-pro", "response-1"))
        self.assertEqual(len(keying), 5)
        self.assertFalse(set(keying) - {"F", "R"})

        raw = dict.fromkeys(FACTOR_ORDER, 1)
        corrected = reverse_score(raw, keying)
        for index, factor in enumerate(FACTOR_ORDER):
            self.assertEqual(corrected[factor], 5 if keying[index] == "R" else 1)

    def test_parser_accepts_json_fences_and_repairs(self):
        expected = {"RE": 1, "DE": 2, "BO": 3, "GU": 4, "VB": 5}
        parsed, error = parse_judge_response(
            "some preamble\n" + "\x60" * 3 + "json\n" + '{"RE":1,"DE":2,"BO":3,"GU":4,"VB":5}' + "\n" + "\x60" * 3
        )
        self.assertIsNone(error)
        self.assertEqual(parsed, expected)

        parsed, error = parse_judge_response(
            "{'RE':1,'DE':2,'BO':3,'GU':4,'VB':5}"
        )
        self.assertIsNone(error)
        self.assertEqual(parsed, expected)

    def test_parser_rejects_missing_or_out_of_range_scores(self):
        parsed, error = parse_judge_response('{"RE":1}')
        self.assertIsNone(parsed)
        self.assertIn("missing_keys", error)
        parsed, error = parse_judge_response(
            '{"RE":1,"DE":2,"BO":3,"GU":4,"VB":6}'
        )
        self.assertIsNone(parsed)
        self.assertEqual(error, "out_of_range:VB")
        parsed, error = parse_judge_response(
            '{"RE":1,"DE":2,"BO":3.7,"GU":4,"VB":5}'
        )
        self.assertIsNone(parsed)
        self.assertEqual(error, "non_integer:BO")


class PipelineTests(unittest.TestCase):
    def test_n_runs_can_expand_without_changing_existing_ids(self):
        first = GenerationConfig("google/gemma-3-4b-it", "vllm", "bfloat16", n_runs=1)
        expanded = GenerationConfig("google/gemma-3-4b-it", "vllm", "bfloat16", n_runs=5)
        self.assertEqual(first.fingerprint, expanded.fingerprint)
        self.assertEqual(response_id(first, "RE-BP01", 1), response_id(expanded, "RE-BP01", 1))

    def test_generation_resume_judging_and_exports(self):
        prompts = load_behavioral_prompts(n_prompts=2)
        config = GenerationConfig(
            model_name="google/gemma-3-4b-it",
            backend="vllm",
            dtype="bfloat16",
            n_runs=2,
            batch_size=2,
        )
        backend = FakeBackend()
        opener = FakeOpener()
        judge = JudgeClient(
            JudgeConfig(
                api_key="test-key",
                thinking="disabled",
                max_attempts=1,
            ),
            opener=opener,
            sleep=lambda _: None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            response_store = JsonlStore(root / "model_responses.jsonl", ("response_id",))
            rating_store = JsonlStore(
                root / "judge_ratings.jsonl",
                ("response_id", "judge_model"),
            )
            responses = run_generation(
                backend,
                config,
                prompts,
                response_store,
                show_progress=False,
            )
            self.assertEqual(len(responses), 4)
            self.assertEqual(backend.calls, 2)
            self.assertTrue(all(response["raw_response"] for response in responses))

            run_generation(
                backend,
                config,
                prompts,
                response_store,
                resume=True,
                show_progress=False,
            )
            self.assertEqual(backend.calls, 2)

            ratings = run_judging(
                judge,
                prompts,
                responses,
                rating_store,
                workers=1,
                show_progress=False,
            )
            self.assertEqual(len(ratings), 4)
            self.assertTrue(all(rating["status"] == "success" for rating in ratings))
            self.assertTrue(all(set(rating["scores"]) == set(FACTOR_ORDER) for rating in ratings))
            self.assertEqual(len(opener.requests), 4)
            self.assertEqual(opener.requests[0][0]["model"], "deepseek-v4-pro")
            self.assertEqual(
                opener.requests[0][0]["thinking"],
                {"type": "disabled"},
            )

            export_results(root, responses, ratings)
            with (root / "results.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("score_RE", rows[0])
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["counts"]["successful_judge_ratings"], 4)

    def test_cli_defaults_keep_value_action_compatible(self):
        value_action = build_value_action_parser().parse_args([])
        self.assertEqual(value_action.backend, "hf")
        self.assertEqual(value_action.dtype, "bfloat16")
        behavioral = build_behavioral_parser().parse_args([])
        self.assertEqual(behavioral.backend, "vllm")
        self.assertEqual(behavioral.dtype, "bfloat16")
        self.assertEqual(behavioral.n_runs, 5)
        self.assertEqual(behavioral.temperature, 1.0)
        self.assertEqual(behavioral.max_new_tokens, 2048)


if __name__ == "__main__":
    unittest.main()
