from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psycho_llm_behavioral.prompts import load_behavioral_prompts
from psycho_llm_behavioral.run import _steering_from_args, build_parser
from psycho_llm_behavioral.runner import GenerationConfig
from psycho_llm_behavioral.steering import (
    SteeringConfig,
    build_steered_messages,
    last_user_turn_mask,
    load_concept_config,
    parse_steering_factors,
    resolve_factor_tokens,
)


class FactorTokenizer:
    vocab_size = 32

    def __init__(self):
        self.encodings = {
            " responsive": [1],
            " deferential": [2, 3],
            " bold": [4],
            "bold": [8],
            " guarded": [5],
            " verbose": [6],
            " custom": [7],
        }
        self.decoded = {
            1: " responsive",
            2: " defer",
            3: "ential",
            4: " bold",
            5: " guarded",
            6: " verbose",
            7: " custom",
            8: "bold",
            9: " direct-id",
        }

    def __len__(self):
        return self.vocab_size

    def encode(self, text, *, add_special_tokens):
        return self.encodings.get(text, [])

    def decode(self, token_ids, *, clean_up_tokenization_spaces):
        return "".join(self.decoded[token_id] for token_id in token_ids)


class SteeringConfigTests(unittest.TestCase):
    def test_factor_parser_accepts_names_codes_and_adjectives(self):
        self.assertEqual(
            parse_steering_factors("Responsiveness,DE,bold,Guardedness,verbose,RE"),
            ("RE", "DE", "BO", "GU", "VB"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown steering factor"):
            parse_steering_factors("helpful")

    def test_prompt_steering_appends_each_adjective_to_final_message(self):
        prompt = next(
            item
            for item in load_behavioral_prompts()
            if item["prompt_id"] == "RE-BP01"
        )
        steering = SteeringConfig(method="prompt", factors=("RE", "DE", "BO"))
        messages = build_steered_messages(prompt, steering)
        self.assertEqual([item["role"] for item in messages], ["user", "assistant", "user"])
        self.assertTrue(
            messages[-1]["content"].endswith(
                "Make the answer responsive. Make the answer deferential. "
                "Make the answer bold."
            )
        )
        self.assertNotIn("Make the answer", prompt["turn2_user"])

    def test_jlens_config_and_cli_require_explicit_layer(self):
        with self.assertRaisesRegex(ValueError, "layer"):
            SteeringConfig(method="jlens", factors=("RE",))

        args = build_parser().parse_args(
            [
                "--steer-method",
                "jlens",
                "--steer-factors",
                "Responsiveness,Boldness",
                "--steer-layer",
                "20",
                "--steer-alpha",
                "1.5",
            ]
        )
        config = _steering_from_args(args)
        self.assertEqual(config.factors, ("RE", "BO"))
        self.assertEqual(config.layer, 20)
        self.assertEqual(config.alpha, 1.5)

    def test_steering_is_part_of_generation_identity(self):
        baseline = GenerationConfig("model", "hf", "float32")
        prompt = GenerationConfig(
            "model",
            "hf",
            "float32",
            steering=SteeringConfig(method="prompt", factors=("RE",)),
        )
        self.assertNotEqual(baseline.fingerprint, prompt.fingerprint)


class JLensTokenTests(unittest.TestCase):
    def test_default_multi_token_adjective_uses_first_jlens_fragment(self):
        tokenizer = FactorTokenizer()
        steering = SteeringConfig(
            method="jlens",
            factors=("RE", "DE", "BO", "GU", "VB"),
            layer=20,
        )
        resolved = resolve_factor_tokens(tokenizer, steering)
        self.assertEqual([item["token_id"] for item in resolved], [1, 2, 4, 5, 6])
        deference = resolved[1]
        self.assertEqual(deference["token_text"], " defer")
        self.assertFalse(deference["is_concept_single_token"])
        self.assertEqual(deference["concept_token_ids"], [2, 3])

    def test_explicit_token_override_must_be_one_token(self):
        tokenizer = FactorTokenizer()
        valid = SteeringConfig(
            method="jlens",
            factors=("DE",),
            layer=20,
            token_overrides=(("DE", " custom"),),
        )
        self.assertEqual(resolve_factor_tokens(tokenizer, valid)[0]["token_id"], 7)

        invalid = SteeringConfig(
            method="jlens",
            factors=("DE",),
            layer=20,
            token_overrides=(("DE", " deferential"),),
        )
        with self.assertRaisesRegex(ValueError, "encoded to 2 tokens"):
            resolve_factor_tokens(tokenizer, invalid)
    def test_config_supports_repeated_variants_all_tokens_and_direct_ids(self):
        payload = {
            "schema_version": 1,
            "name": "multi-concept-test",
            "factors": {
                "BO": {
                    "prompt_adjective": "bold",
                    "concepts": [
                        {"text": "bold"},
                        {"text": "bold", "prepend_space": False},
                        {
                            "text": "deferential",
                            "token_selection": "all",
                        },
                        {"token_id": 9, "label": "model-specific direction"},
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "concepts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            concept_config = load_concept_config(path)
            steering = SteeringConfig(
                method="jlens",
                factors=("BO",),
                layer=20,
                concept_config=concept_config,
            )
            with self.assertWarnsRegex(UserWarning, "tokenizer-specific"):
                resolved = resolve_factor_tokens(FactorTokenizer(), steering)

            self.assertEqual(
                [item["token_id"] for item in resolved],
                [4, 8, 2, 3, 9],
            )
            self.assertEqual(
                [item["concept_item_index"] for item in resolved],
                [0, 1, 2, 2, 3],
            )
            self.assertEqual(resolved[0]["encoded_text"], " bold")
            self.assertEqual(resolved[1]["encoded_text"], "bold")
            self.assertEqual(
                [item["selected_token_index"] for item in resolved[2:4]],
                [0, 1],
            )
            self.assertIn(concept_config.sha256[:8], steering.slug)
            self.assertEqual(
                steering.public_dict()["concept_config"]["source_path"],
                str(path.resolve()),
            )

            args = build_parser().parse_args(
                [
                    "--steer-method",
                    "jlens",
                    "--steer-factors",
                    "bold",
                    "--steer-layer",
                    "20",
                    "--steer-concept-config",
                    str(path),
                ]
            )
            from_cli = _steering_from_args(args)
            self.assertEqual(from_cli.concept_config.sha256, concept_config.sha256)
            self.assertEqual(from_cli.factors, ("BO",))


    def test_vllm_mask_selects_only_final_user_turn_per_sequence(self):
        prefix = [90, 91]
        suffix = [92]
        first = [1, 90, 91, 10, 11, 92, 50]
        second = [1, 90, 91, 20, 92, 60, 90, 91, 30, 31, 92, 70]
        input_ids = first + second
        positions = list(range(len(first))) + list(range(len(second)))
        mask = last_user_turn_mask(input_ids, positions, prefix, suffix)
        selected = [token for token, is_selected in zip(input_ids, mask) if is_selected]
        self.assertEqual(selected, [10, 11, 30, 31])


if __name__ == "__main__":
    unittest.main()
