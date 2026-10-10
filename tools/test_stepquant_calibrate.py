#!/usr/bin/env python3
"""CPU calibration tests: global optimum, pivots/budget and engine trace layout."""
import itertools
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from stepquant_calibrate import allocate, calibrate, fit, impact_factors, lifetime, read_statistics


class CalibrationTest(unittest.TestCase):
    def test_dp_global_optimum(self):
        rng = np.random.default_rng(41)
        bits = (2, 4, 6, 8)
        for n in range(1, 5):
            cost = rng.uniform(0, 1, (n, 4))
            for budget in range(2*n, 8*n+1, 2):
                choice = allocate(cost, bits, budget)
                actual = sum(cost[i, bits.index(int(b))] for i, b in enumerate(choice))
                optimum = min(sum(cost[i, bits.index(b)] for i, b in enumerate(c))
                              for c in itertools.product(bits, repeat=n) if sum(c) <= budget)
                self.assertAlmostEqual(actual, optimum, places=12)
        with self.assertRaises(ValueError): allocate(np.zeros((3,4)), bits, 5)

    def test_pivots_and_budget(self):
        rng = np.random.default_rng(7)
        stats = {i: dict(omega=rng.uniform(.1,2,(4,128)), decay=np.array([0,-.01,-.1,-3]),
                        samples=[rng.normal(0,.1,(4,128,128)).astype(np.float32)], tokens=8, heads=4) for i in (0,2)}
        for nominal in (4,6):
            plans = calibrate(stats, nominal, 2, 128)
            bits = np.concatenate([p[0] for p in plans.values()])
            self.assertEqual(int((bits == 16).sum()), 2)
            self.assertLessEqual(int(bits[bits != 16].sum()) + 8*2, nominal*8)
            for _, w in plans.values():
                self.assertTrue(np.isfinite(w).all()); self.assertTrue((w > 0).all())
                np.testing.assert_allclose(np.log(w).mean(-1),0,atol=1e-7)
        np.testing.assert_allclose(lifetime(np.array([0., -100]),128), [128,1])
        for b in (2,4,6,8): self.assertTrue(np.isfinite(fit(np.zeros((1,128,128),np.float32),np.ones((1,128),np.float32),b)).all())
        with self.assertRaises(ValueError): calibrate(stats,4,7)

    def test_trace_layout_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'layer-0.bin'
            H, HK = 4, 2
            qk = np.zeros((2,HK,128),np.float32); qk[0,:,0]=1; qk[1,:,1]=1
            state = np.arange(H*128*128,dtype=np.float32).reshape(128,H,128)*1e-5
            raw = struct.pack('<7i',0x53515452,1,H,HK,128,1,2)+struct.pack('<i',1)
            raw += qk.tobytes() + np.full(H,-.1,np.float32).tobytes() + np.full(H,.5,np.float32).tobytes()+state.tobytes()
            path.write_bytes(raw)
            stat = read_statistics(path)
            self.assertEqual(stat['tokens'],1)
            np.testing.assert_array_equal(stat['samples'][0],state.transpose(1,0,2))
            self.assertTrue((impact_factors(stat['omega']) > 0).all())
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError): read_statistics(path)


if __name__ == '__main__': unittest.main()
