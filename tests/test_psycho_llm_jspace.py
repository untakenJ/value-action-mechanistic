from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from jlens_experiments.common import gradient_pursuit_decompose
from psycho_llm_behavioral.inspect_jspace import (
    CSV_FIELDNAMES,
    _default_csv_path,
    _format_jspace_tokens,
    _format_lens_tokens,
    _write_csv,
    build_parser,
    csv_rows_from_records,
    factorized_gradient_pursuit,
    resolve_layers,
    top_token_records,
)


class FakeTokenizer:
    def decode(self, token_ids, **_kwargs):
        return f"<{token_ids[0]}>"


class LayerSelectionTests(unittest.TestCase):
    def test_resolve_all_explicit_and_nearest_percentage(self):
        fitted = [0, 4, 8, 12]
        self.assertEqual(resolve_layers("all", fitted, 13), fitted)
        self.assertEqual(resolve_layers("8", fitted, 13), [8])
        self.assertEqual(resolve_layers("8,4", fitted, 13), [4, 8])
        self.assertEqual(resolve_layers("65%", fitted, 13), [8])
        self.assertEqual(resolve_layers("4,65%", fitted, 13), [4, 8])

    def test_resolve_multiple_percentages_and_workspace(self):
        fitted = [0, 4, 8, 12]
        self.assertEqual(resolve_layers("50%,65%", fitted, 13), [4, 8])
        self.assertEqual(
            resolve_layers("workspace", fitted, 13),
            [0, 4, 8, 12],
        )

    def test_rejects_unfitted_and_invalid_layers(self):
        with self.assertRaisesRegex(ValueError, "not fitted"):
            resolve_layers("5", [0, 4, 8], 9)
        with self.assertRaisesRegex(ValueError, "between 0% and 100%"):
            resolve_layers("101%", [0, 4, 8], 9)


class JSpaceDecompositionTests(unittest.TestCase):
    def test_factorized_pursuit_matches_existing_dense_implementation(self):
        generator = torch.Generator().manual_seed(17)
        unembedding = torch.randn(19, 7, generator=generator)
        jacobian = torch.randn(7, 7, generator=generator)
        activation = torch.randn(7, generator=generator)
        dictionary = unembedding @ jacobian

        dense = gradient_pursuit_decompose(activation, dictionary, k=5)
        factorized = factorized_gradient_pursuit(
            activation,
            unembedding,
            jacobian,
            k=5,
            chunk_size=4,
        )

        self.assertEqual(factorized.token_ids, dense.token_ids)
        self.assertEqual(len(factorized.coefficients), len(dense.coefficients))
        for actual, expected in zip(
            factorized.coefficients,
            dense.coefficients,
        ):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertAlmostEqual(
            factorized.variance_fraction,
            dense.variance_fraction,
            places=5,
        )
        self.assertAlmostEqual(
            factorized.residual_variance_fraction,
            dense.residual_variance_fraction,
            places=5,
        )

    def test_zero_activation_has_no_active_jspace_coordinates(self):
        result = factorized_gradient_pursuit(
            torch.zeros(3),
            torch.eye(3),
            torch.eye(3),
            k=2,
            chunk_size=1,
        )
        self.assertEqual(result.token_ids, ())
        self.assertEqual(result.coefficients, ())
        self.assertEqual(result.variance_fraction, 0.0)
        self.assertEqual(result.residual_variance_fraction, 0.0)

    def test_top_token_records_preserve_rank_id_text_and_logit(self):
        rows = top_token_records(
            FakeTokenizer(),
            torch.tensor([0.1, 3.0, 2.0]),
            2,
        )
        self.assertEqual([row["rank"] for row in rows], [1, 2])
        self.assertEqual([row["token_id"] for row in rows], [1, 2])
        self.assertEqual([row["token_text"] for row in rows], ["<1>", "<2>"])

    def test_cli_defaults_follow_paper_inventory_convention(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.layers, "all")
        self.assertEqual(args.jspace_layers, "65%")
        self.assertEqual(args.jspace_k, 25)
        self.assertTrue(args.sparse_decomposition)
        self.assertFalse(args.no_csv)
        self.assertEqual(
            _default_csv_path(args.output),
            Path("outputs/psycho_llm_behavioral/jspace_last_token.csv"),
        )


class CsvExportTests(unittest.TestCase):
    def test_csv_rows_include_lens_and_jspace_summaries(self):
        rows = csv_rows_from_records(
            [
                {
                    "prompt_id": "BO-BP01",
                    "layers": [
                        {
                            "layer": 21,
                            "top_lens_tokens": [
                                {"token_text": "Princess"},
                                {"token_text": "Okay"},
                            ],
                            "jspace_decomposition": {
                                "tokens": [
                                    {
                                        "token_text": "winds",
                                        "coefficient": 248.9,
                                    }
                                ]
                            },
                        },
                        {
                            "layer": 22,
                            "top_lens_tokens": [{"token_text": "Queen"}],
                            "jspace_decomposition": None,
                        },
                    ],
                }
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prompt_id"], "BO-BP01")
        self.assertEqual(rows[0]["layer"], "21")
        self.assertEqual(rows[0]["j_lens_tokens"], "Princess | Okay")
        self.assertEqual(rows[0]["jspace_tokens"], "winds:248.9")
        self.assertEqual(rows[1]["jspace_tokens"], "")

    def test_write_csv_round_trip(self):
        records = [
            {
                "prompt_id": "DE-BP03",
                "layers": [
                    {
                        "layer": 8,
                        "top_lens_tokens": [{"token_text": "Wow"}],
                        "jspace_decomposition": None,
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            _write_csv(path, records)
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0].keys()), list(CSV_FIELDNAMES))
        self.assertEqual(rows[0]["prompt_id"], "DE-BP03")
        self.assertEqual(rows[0]["layer"], "8")
        self.assertEqual(rows[0]["j_lens_tokens"], "Wow")
        self.assertEqual(rows[0]["jspace_tokens"], "")

    def test_token_formatters_join_with_pipe(self):
        self.assertEqual(
            _format_lens_tokens([{"token_text": "a"}, {"token_text": "b"}]),
            "a | b",
        )
        self.assertEqual(
            _format_jspace_tokens(
                {"tokens": [{"token_text": "x", "coefficient": 1.2345}]}
            ),
            "x:1.234",
        )
        self.assertEqual(_format_jspace_tokens(None), "")


if __name__ == "__main__":
    unittest.main()
