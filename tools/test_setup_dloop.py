"""DLoop setup, gate semantics and reversible config tests (no GPU or downloads)."""
import argparse
import copy
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import dloop_setup as D
import setup as S


def options(*argv):
    p = argparse.ArgumentParser()
    D.arguments(p)
    p.add_argument('--yes', action='store_true')
    return p.parse_args(argv)


BASE = {'args': ['--native', 'model.gguf', '--mtp', 'mtp/rt', '--spec', '4', '--spec-min-p', '0.5', '--suffix-draft', '12'],
        'parallel': 1, 'sampling': {'temperature': 0.7}, 'env': {'CUSTOM': '1'}}
ON = {'enabled': True, 'block_size': 3, 'max_loops': 2, 'gate': -0.5}


class Setup(unittest.TestCase):
    def test_default_off_preserves_config(self):
        c = copy.deepcopy(BASE)
        answer = D.choose(options('--yes'), S.ask, lambda _: None)
        self.assertFalse(answer['enabled'])
        self.assertEqual(D.apply(c, answer), BASE)

    def test_questions_follow_enable_and_limit_choices(self):
        ask = mock.Mock(side_effect=['y', '4', '1', '-1'])
        c = D.choose(options(), ask, lambda _: None)
        self.assertEqual(c, {**ON, 'block_size': 4, 'max_loops': 1, 'gate': -1})
        self.assertEqual(ask.call_args_list[2].args[1], ['1'])

    def test_cli_parameters_imply_on(self):
        ask = mock.Mock(side_effect=lambda q, c, d, y: d)
        c = D.choose(options('--yes', '--dloop-block-size', '2', '--dloop-gate=-1', '--dloop-max-loops', '3'), ask, lambda _: None)
        self.assertEqual(c, {**ON, 'block_size': 2, 'max_loops': 3, 'gate': -1})
        self.assertFalse(ask.called)

    def test_invalid_parameters(self):
        for change in ({'block_size': 0}, {'block_size': 8}, {'max_loops': 0}, {'max_loops': 3},
                       {'gate': 0.1}, {'gate': math.nan}, {'gate': -math.inf}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                D.validate({**ON, **change})
        with self.assertRaises(ValueError):
            D.choose(options('--yes', '--dloop', 'off', '--dloop-gate=-1'), S.ask, lambda _: None)

    def test_enable_reconfigure_disable_restores_only_managed_flags(self):
        c = D.apply(copy.deepcopy(BASE), ON)
        previous = copy.deepcopy(c['dloop']['previous'])
        self.assertEqual(D.value(c['args'], '--spec'), '7')
        self.assertEqual(D.value(c['args'], '--spec-min-p'), '0')
        c['args'] += ['--pool-workers', '6']
        D.apply(c, {**ON, 'block_size': 2, 'max_loops': 3})
        self.assertEqual(c['dloop']['previous'], previous)
        D.apply(c, {'enabled': False})
        self.assertNotIn('dloop', c)
        self.assertEqual(c, {**BASE, 'args': BASE['args'][:4] + ['--suffix-draft', '12', '--pool-workers', '6',
                           '--spec', '4', '--spec-min-p', '0.5']})

    def test_incompatible_installs_unchanged(self):
        variants = [{**BASE, 'parallel': 2}, {**BASE, 'args': ['--native', 'x', '--dflash', 'y']},
                    {**BASE, 'args': BASE['args'] + ['--pipeline-windows', '2']},
                    {**BASE, 'args': BASE['args'] + ['--batch', '2']},
                    {**BASE, 'args': BASE['args'] + ['--lookup-chain', '2']}]
        for variant in variants:
            c = copy.deepcopy(variant)
            with self.assertRaises(ValueError):
                D.apply(c, ON)
            self.assertEqual(c, variant)

    def test_disabled_metadata_is_not_carried_over(self):
        c = copy.deepcopy(BASE)
        old = D.apply(copy.deepcopy(BASE), ON)
        S.carry_over(old, c)
        self.assertNotIn('dloop', c)

    def test_capability_needs_actual_advertisement(self):
        for code, output, expected in [(0, '--dloop-block-size', True), (0, '', False), (1, '--dloop-block-size', False)]:
            self.assertEqual(D.capability('strata', lambda *a, **k: subprocess.CompletedProcess([], code, output, '')), expected)


class Gate(unittest.TestCase):
    def test_native_policy_and_latest_block_reset(self):
        # Exercise the same header the CUDA/HIP/SYCL host code uses, rather than a Python copy.
        source = r'''
#include "strata/spec/dloop.hpp"
#include <cassert>
#include <limits>
int main() {
    strata::spec::DLoopConfig c;
    assert(c.error().empty());
    float good[] = {1, 1, 1}, bad[] = {.1f, .1f, .1f};
    assert(c.extend(good, 3));
    assert(!c.extend(bad, 3));
    assert(!c.extend(good, 2));
    c.gate = 0; assert(c.extend(good, 3)); // equality continues
    float invalid[] = {0, 1, 1}; assert(!c.extend(invalid, 3));
    invalid[0] = std::numeric_limits<float>::quiet_NaN(); assert(!c.extend(invalid, 3));
    invalid[0] = 1.1; assert(!c.extend(invalid, 3));
    c.block_size = 2; c.max_loops = 3; c.gate = -0.5;
    float blocks[] = {.9f, .9f, .8f, .8f, .9f, .9f};
    // The second block passes on its own; a cumulative gate would wrongly stop before block three.
    assert(c.extend(blocks + 2, 2)); assert(c.extend(blocks + 4, 2));
    c.max_loops = 4; assert(!c.error().empty());
    c.block_size = 0; assert(!c.error().empty());
    c.block_size = 7; c.max_loops = 1; assert(c.error().empty());
    c.gate = std::numeric_limits<double>::infinity(); assert(!c.error().empty());
}
'''
        with tempfile.TemporaryDirectory() as temp:
            cpp, exe = Path(temp) / 'gate.cpp', Path(temp) / 'gate'
            cpp.write_text(source)
            subprocess.run(['c++', '-std=c++20', '-I', str(ROOT / 'include'), str(cpp), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    unittest.main()
