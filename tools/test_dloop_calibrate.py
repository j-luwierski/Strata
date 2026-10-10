import copy
import contextlib
import io
import json
import tempfile
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import dloop_calibrate as C, dloop_setup as D
from tools.test_setup_dloop import BASE, ON
import setup as S


class FakeEngine:
    def __init__(self, args, closed, drift=False, fail=False):
        self.args, self.closed, self.drift, self.fail = args, closed, drift, fail
        self.info = {'dloop_cache_budget': 100}
        self.last = {}
    def generate(self, ids, n, sampling, cancel):
        if self.fail: raise RuntimeError('GPU failure')
        enabled = '--dloop' in self.args
        self.last = {'decode_ms': n * (5 if enabled else 10)}
        return iter([1 if not (enabled and self.drift) else 2] * n)
    def close(self): self.closed.append(self)


class Calibration(unittest.TestCase):
    def test_rejects_fast_quality_mismatch(self):
        rows = {'off': [{'tok_s': 10, 'same_tokens': True}], 'fast': [{'tok_s': 50, 'same_tokens': False}]}
        choice, rates = C.select(rows, {'off': {'enabled': False}, 'fast': ON})
        self.assertFalse(choice['enabled']); self.assertNotIn('fast', rates)

    def test_three_percent_margin(self):
        for rate, enabled in [(10.2, False), (10.4, True)]:
            choice, _ = C.select({'off': [{'tok_s': 10, 'same_tokens': True}],
                                 'on': [{'tok_s': rate, 'same_tokens': True}]}, {'off': {'enabled': False}, 'on': ON})
            self.assertEqual(choice['enabled'], enabled)

    def test_stable_baseline_required(self):
        with self.assertRaises(RuntimeError):
            C.select({'off': [{'tok_s': 10, 'same_tokens': False}]}, {'off': {'enabled': False}})

    def test_measure_restores_config_and_closes_engines(self):
        for drift, winner in [(False, 'b3-n2-g-0.5'), (True, 'off')]:
            cfg = D.apply(copy.deepcopy(BASE), ON)
            cfg['args'] += ['--expert-cache', '100']
            original = copy.deepcopy(cfg); closed = []
            with mock.patch.object(C.CAL, 'engine_args', side_effect=lambda c: c['args']):
                result = C.measure(cfg, [[11], [12]], lambda args: FakeEngine(args, closed, drift),
                                   say=lambda _: None, candidates=[], repeats=2)
            self.assertEqual(result['report']['winner'], winner)
            self.assertEqual(len(closed), 4)
            self.assertEqual(cfg, original)

    def test_auto_cache_probe_and_failed_candidates(self):
        cfg = D.apply(copy.deepcopy(BASE), ON); closed = []
        with mock.patch.object(C.CAL, 'engine_args', side_effect=lambda c: c['args']):
            result = C.measure(cfg, [[11]], lambda args: FakeEngine(args, closed, fail='--dloop' in args),
                               say=lambda _: None, candidates=[], repeats=1)
        self.assertEqual(result['report']['winner'], 'off')
        self.assertEqual(len(closed), 3)
        self.assertEqual(C.CAL.arg_value(result['report']['base_args'], '--expert-cache'), '100')


class SavedCalibration(unittest.TestCase):
    def config(self, root):
        cfg = D.apply(copy.deepcopy(BASE), ON)
        path = Path(root) / 'strata-test.json'
        path.write_text(json.dumps(cfg))
        return path, cfg

    def test_failed_measurement_leaves_config_byte_identical(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.config(root)
            before = path.read_bytes()
            with mock.patch.object(C, 'run', side_effect=RuntimeError('GPU failure')), contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(S.calibrate_dloop_config(path))
            self.assertEqual(path.read_bytes(), before)

    def test_off_winner_restores_flags_and_saves_report(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.config(root)
            result = {'choice': {'enabled': False}, 'report': {'winner': 'off', 'base_args': []}}
            with mock.patch.object(C, 'run', return_value=result), contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(S.calibrate_dloop_config(path))
            cfg = json.loads(path.read_text())
            self.assertNotIn('dloop', cfg)
            self.assertEqual(D.value(cfg['args'], '--spec-min-p'), '0.5')
            self.assertEqual(cfg['sampling'], BASE['sampling'])
            self.assertEqual(cfg['dloop_calibration'], result['report'])

    def test_on_winner_saves_measured_cache_budget(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.config(root)
            result = {'choice': ON, 'report': {'winner': C.key(ON), 'base_args': ['--expert-cache', '90']}}
            with mock.patch.object(C, 'run', return_value=result), contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(S.calibrate_dloop_config(path))
            cfg = json.loads(path.read_text())
            self.assertEqual(D.value(cfg['args'], '--expert-cache'), '90')
            self.assertEqual(cfg['dloop']['previous']['--spec'], '4')


if __name__ == '__main__': unittest.main()
