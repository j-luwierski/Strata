#!/usr/bin/env python3
"""CPU-only exporter validation; no GPU, torch, downloads or model needed."""
import copy
import unittest
from stepquant_plan import export_plan


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def tolist(self):
        return self.values


def artifact():
    return {"format_version": 2, "plans": {
        "model.layers.0.linear_attn": {"architecture": "gdn", "value_group_size": 32,
            "bits": Tensor([[4]*128, [16]*128]), "impact": Tensor([[1.]*128, [2.]*128])}}}


class PlanTest(unittest.TestCase):
    def test_export(self):
        text = export_plan(artifact(), 2, 2, 2)
        rows = text.splitlines()
        self.assertEqual(rows[:3], ["STRATA_STEPQUANT 1 128 2 1", "0", "4 16"])
        self.assertEqual(len(rows[3].split()), 256)

    def test_reject_foreign_or_incomplete_artifacts(self):
        wrong_version = artifact()
        wrong_version["format_version"] = 1
        missing = artifact()
        missing["plans"].clear()
        qsa = artifact()
        qsa["plans"]["model.layers.1.linear_attn"] = qsa["plans"].pop("model.layers.0.linear_attn")
        duplicate = artifact()
        duplicate["plans"]["0"] = duplicate["plans"]["model.layers.0.linear_attn"]
        for bad in (wrong_version, missing, qsa, duplicate):
            with self.assertRaises(ValueError):
                export_plan(bad, 2, 2, 2)
        with self.assertRaises(ValueError):
            export_plan(artifact(), 2, 2, 3)

    def test_invalid_precision_impact_and_architecture(self):
        for field, value in (("architecture", "kda"), ("value_group_size", 64),
                             ("bits", Tensor([[3]*128]*2)),
                             ("bits", Tensor([[4]*127+[6], [4]*128])),
                             ("impact", Tensor([[float("nan")]*128]*2)),
                             ("impact", Tensor([[0.]*128]*2))):
            a = copy.deepcopy(artifact())
            a["plans"]["model.layers.0.linear_attn"][field] = value
            with self.assertRaises(ValueError):
                export_plan(a, 2, 2, 2)


if __name__ == "__main__":
    unittest.main()
