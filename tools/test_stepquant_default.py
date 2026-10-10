#!/usr/bin/env python3
"""Check that changed default-path sources equal origin/main with the opt-in blocks disabled.

This checks source preservation, not full-model output parity. Run from the repo root.
"""
from pathlib import Path
import subprocess
files=['src/core/layer.cpp','src/core/verify.cpp','src/prefill/prefill.cpp','src/program/generate.cpp','src/core/session.cpp','src/core/conversation_state.cpp','include/strata/core/layer.hpp']
files += ['sycl/'+name for name in files if Path('sycl/'+name).is_file()]
for name in files:
 lines=Path(name).read_text().splitlines(keepends=True); result=[]; active=True; stack=[]
 for line in lines:
  text=line.strip()
  if text=='#ifdef STRATA_ENABLE_STEPQUANT': stack.append(active);active=False;continue
  if stack and text=='#else': active=stack[-1] and not active;continue
  if stack and text=='#endif': active=stack.pop();continue
  if active: result.append(line)
 original=subprocess.check_output(['git','show','origin/main:'+name]).decode()
 if ''.join(result)!=original:
  import difflib
  print(''.join(difflib.unified_diff(original.splitlines(keepends=True),result)))
  raise AssertionError(name)
 print(name+': default source byte-identical to origin/main')
