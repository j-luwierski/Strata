"""Installer STEPQuant choices, config safety and build caching (no GPU/downloads)."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import struct
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup as S
from tools import stepquant_setup as SQ


def options(*args):
    p = argparse.ArgumentParser()
    p.add_argument('--yes', action='store_true')
    SQ.arguments(p)
    a = p.parse_args(args)
    SQ.validate_arguments(a, p)
    return a


def plan(path, bits=6):
    path.write_text('STRATA_STEPQUANT 1 128 1 1\n0\n' + str(bits) + '\n' + ' '.join(['1']*128) + '\n')
    return path


class Choices(unittest.TestCase):
    def setUp(self):
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
    def tearDown(self):
        self.quiet.__exit__(None, None, None)
    def test_yes_keeps_opt_in_off(self):
        self.assertIsNone(SQ.choose(S, options('--yes')))
    def test_on_asks_nested_choices(self):
        answers = iter(['y', 'calibrate', '4', '16', '512', 'default'])
        with mock.patch.object(S, 'ask', side_effect=lambda *a: next(answers)):
            choice = SQ.choose(S, options())
        self.assertEqual((choice['bits'], choice['pivots'], choice['horizon']), (4,16,512))
        self.assertIsNone(choice['plan'])
    def test_flags_enable_calibration_without_questions(self):
        c = SQ.choose(S, options('--yes','--stepquant-bits','4','--stepquant-pivots','0','--stepquant-horizon','512'))
        self.assertEqual((c['bits'],c['pivots'],c['horizon']), (4,0,512))
    def test_existing_plan_reused_with_settings(self):
        with tempfile.TemporaryDirectory() as d:
            old = {'stepquant_plan':str(plan(Path(d)/'old.plan')), 'stepquant':dict(bits=4,pivots=16,horizon=512)}
            c = SQ.choose(S, options('--yes','--stepquant','on'), old)
            self.assertEqual(c['plan'],old['stepquant_plan'])
            self.assertEqual(c['bits'],4)
            fresh = SQ.choose(S, options('--yes','--stepquant-recalibrate'), old)
            self.assertIsNone(fresh['plan'])
            changed = SQ.choose(S, options('--yes','--stepquant-bits','6'), old)
            self.assertIsNone(changed['plan'])
    def test_off_conflicts_rejected(self):
        for args in [('--stepquant-bits','4'), ('--stepquant-recalibrate',), ('--stepquant-pivots','-1'), ('--stepquant-horizon','0')]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options('--stepquant','off',*args)


class Plans(unittest.TestCase):
    def test_import_then_disable_removes_raw_override(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            d=Path(d); source=plan(d/'source.plan')
            cfg={'args':['--stepquant-plan','old.plan']}
            choice=dict(plan=str(source),bits=6,pivots=32,horizon=2048,corpus=str(SQ.CORPUS))
            SQ.apply(S,cfg,choice,d/'installed')
            self.assertEqual(Path(cfg['stepquant_plan']).read_bytes(),source.read_bytes())
            self.assertNotIn('--stepquant-plan',cfg['args'])
            old=dict(cfg)
            SQ.apply(S,cfg,None,d/'installed')
            S.carry_over(old,cfg)
            self.assertNotIn('stepquant_plan',cfg)
            self.assertNotIn('stepquant',cfg)
    def test_invalid_import_cannot_replace_existing_plan(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); valid=plan(d/'valid.plan'); saved=valid.read_bytes()
            bad=plan(d/'bad.plan',3)
            with self.assertRaises(ValueError):
                SQ.apply(S,{'args':[]},dict(plan=str(bad)),d)
            self.assertEqual(valid.read_bytes(),saved)
    def test_nonfinite_duplicate_and_truncated_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p'
            for data in ['STRATA_STEPQUANT 1 128 0 1',
                         'STRATA_STEPQUANT 1 128 1 1\n0\n6\n'+' '.join(['nan']*128),
                         'STRATA_STEPQUANT 1 128 1 1\n0\n6\n'+' '.join(['0']*128)]:
                p.write_text(data)
                with self.assertRaises(ValueError): SQ.validate_plan(p)
    def test_failed_trace_leaves_old_plan_on_disk(self):
        # Config persistence belongs to setup and happens only after apply succeeds.
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            d=Path(d); p=plan(d/'old.plan'); saved=p.read_bytes()
            corpus=d/'text'; corpus.write_text('text')
            cfg=dict(exe='engine',args=['--native','model.gguf'],stepquant_plan=str(p))
            tk=mock.Mock(); tk.encode.return_value=list(range(20))
            with mock.patch('tools.strata_tokenizer.Tokenizer.from_gguf', return_value=tk), \
                 mock.patch.object(SQ,'model_geometry',return_value=(48,set(range(36)))), \
                 mock.patch.object(S,'run',side_effect=RuntimeError('GPU failed')):
                with self.assertRaises(RuntimeError):
                    SQ.apply(S,cfg,dict(plan=None,bits=6,pivots=32,horizon=2048,corpus=str(corpus)),d)
            self.assertEqual(p.read_bytes(),saved)
            self.assertEqual(cfg['stepquant_plan'],str(p))

    def test_trace_calibration_writes_completed_plan_and_metadata(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            d=Path(d); corpus=d/'text'; corpus.write_text('a representative calibration text')
            cfg=dict(exe='engine',args=['--native','model.gguf','--prefill','auto','--spec','4','--batch','2'],gpu=[0,1])
            tk=mock.Mock(); tk.encode.return_value=list(range(20))
            def run(command,**kwargs):
                self.assertNotIn('--batch',command)
                self.assertEqual(command[command.index('--spec')+1],'2')
                self.assertEqual(command[command.index('--gpu')+1],'0')
                trace=Path(command[command.index('--stepquant-trace')+1]); trace.mkdir()
                with (trace/'layer-0.bin').open('wb') as f:
                    f.write(struct.pack('<7i',0x53515452,1,1,1,128,8,512))
                    for i in range(19):
                        sample=(i+1)%8==0
                        f.write(struct.pack('<i',sample))
                        f.write(bytes(2*128*4))
                        f.write(struct.pack('<2f',-1,0.5))
                        if sample: f.write(bytes(128*128*4))
            with mock.patch('tools.strata_tokenizer.Tokenizer.from_gguf',return_value=tk), \
                 mock.patch.object(SQ,'model_geometry',return_value=(1,{0})), \
                 mock.patch.object(S,'run',side_effect=run):
                SQ.apply(S,cfg,dict(plan=None,bits=6,pivots=0,horizon=512,corpus=str(corpus)),d/'installed')
            target=Path(cfg['stepquant_plan'])
            self.assertEqual(SQ.validate_plan(target),(1,{0}))
            report=json.loads(target.with_suffix('.plan.json').read_text())
            self.assertEqual(report['tokens'],19)
            self.assertEqual(sum(report['head_counts'].values()),1)
            self.assertFalse(list((d/'installed').glob('stepquant-*')))


class Installed(unittest.TestCase):
    def test_off_via_main_keeps_user_settings_and_removes_plan(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); cfg_path=root/'strata-model.json'
            old=dict(exe='engine',args=['--native','model.gguf','--stepquant-plan','old.plan'],
                     stepquant_plan='old.plan',stepquant=dict(bits=6),sampling=dict(temperature=0.4),port=8111)
            cfg_path.write_text(json.dumps(old))
            with mock.patch.object(S,'ROOT',root), mock.patch.object(S,'data_folder',return_value=(root/'data',[])), \
                 mock.patch.object(S,'installed_configs',return_value=[cfg_path]), \
                 mock.patch.object(S,'stepquant_engine') as build, mock.patch.object(S,'start') as start, \
                 mock.patch.object(sys,'argv',['setup.py','--yes','--no-start','--stepquant','off']):
                self.assertEqual(S.main(),0)
            cfg=json.loads(cfg_path.read_text())
            self.assertNotIn('stepquant_plan',cfg)
            self.assertNotIn('--stepquant-plan',cfg['args'])
            self.assertEqual(cfg['sampling'],old['sampling'])
            self.assertEqual(cfg['port'],8111)
            build.assert_not_called(); start.assert_not_called()
    def test_invalid_plan_via_main_leaves_config_byte_identical(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); cfg_path=root/'strata-model.json'
            cfg_path.write_text(json.dumps(dict(exe='engine',args=[])))
            before=cfg_path.read_bytes()
            bad=plan(root/'bad.plan',3)
            with mock.patch.object(S,'ROOT',root), mock.patch.object(S,'data_folder',return_value=(root/'data',[])), \
                 mock.patch.object(S,'installed_configs',return_value=[cfg_path]), \
                 mock.patch.object(sys,'argv',['setup.py','--yes','--no-start','--stepquant-plan',str(bad)]):
                with self.assertRaises(SystemExit): S.main()
            self.assertEqual(cfg_path.read_bytes(),before)


class Build(unittest.TestCase):
    def test_cuda12_migration_keeps_stepquant_build_support(self):
        cfg=dict(exe='engine',args=[],stepquant_plan='model.plan')
        card=dict(index=0,arch=61)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(S,'gpu_info',return_value=card), \
             mock.patch.object(S,'get_llama_cpp',return_value='llama'), \
             mock.patch.object(S,'build_engine',return_value=Path('/tmp/engine-cuda12')) as build, \
             mock.patch.object(S,'get_cuda12_engine') as prebuilt, mock.patch.object(S,'engine_lib_dirs',return_value=[]), \
             mock.patch.object(S,'write_config'):
            S.use_cuda12([card],Path('cfg.json'),cfg,True)
        self.assertEqual(build.call_args.kwargs,dict(toolkit=12,stepquant=True))
        self.assertEqual(cfg['cuda'],12)
        prebuilt.assert_not_called()

    def test_cached_off_engine_rebuilt_on_and_capability_kept_on_update(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); (root/'CMakeLists.txt').write_text('project(strata VERSION 0.1.99)'); eng=root/'engine'; eng.mkdir(); (eng/S.EXE).write_text('old')
            stamp=eng/'BUILD.json'
            stamp.write_text(json.dumps(dict(source='local',archs=[89],src='same',vision='none')))
            def build(src,bdir,target,defs,*a):
                self.assertIn('-DSTRATA_ENABLE_STEPQUANT=ON',defs)
                bdir.mkdir(exist_ok=True); (bdir/S.EXE).write_text('new')
            with mock.patch.object(S,'ROOT',root), mock.patch.object(S,'source_hash',return_value='same'), \
                 mock.patch.object(S,'cpu_info',return_value=('cpu',{})), mock.patch.object(S,'cpu_floor',return_value=''), \
                 mock.patch.object(S,'install_build_tools',return_value=('/usr/local/cuda/bin/nvcc',None)), \
                 mock.patch.object(S,'cmake_build',side_effect=build) as compile:
                S.build_engine(dict(arch=89),'none',True,root,stepquant=True)
                self.assertTrue(json.loads(stamp.read_text())['stepquant'])
                S.build_engine(dict(arch=89),'none',True,root)
                self.assertEqual(compile.call_count,1)
                with mock.patch.object(S,'source_hash',return_value='changed'):
                    S.build_engine(dict(arch=89),'none',True,root)
                self.assertEqual(compile.call_count,2)


if __name__=='__main__':
    unittest.main()
