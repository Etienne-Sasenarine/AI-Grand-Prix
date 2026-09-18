import importlib.util
from pathlib import Path
import sys
import types
import unittest

fake_msp = types.ModuleType("msp")
fake_msp.MSPError = type("MSPError", (Exception,), {})
fake_msp.MSPTimeout = type("MSPTimeout", (Exception,), {})
fake_msp.MSPLink = object
sys.modules.setdefault("msp", fake_msp)
spec = importlib.util.spec_from_file_location("hover_logger", Path(__file__).with_name("hover_logger.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def row(t, throttle=1260, armed=True, vario=0.01):
    return dict(t_s=t, armed=armed, throttle_us=throttle, roll_deg=1, pitch_deg=-1,
                gx_dps=1, gy_dps=2, gz_dps=1, vario_m_s=vario)


class HoverLoggerTests(unittest.TestCase):
    def test_requires_three_separate_trials(self):
        rows = []
        for trial in range(3):
            rows.extend(row(trial * 5 + i / 10, 1250 + trial * 10) for i in range(31))
            rows.append(row(trial * 5 + 3.2, armed=False))
        result = mod.summarize(rows, 3)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["hover_throttle_us"], 1260)
        self.assertAlmostEqual(result["hover_thrust_normalized"], 0.26)

    def test_rejects_motion_and_single_period(self):
        moving = [dict(row(i / 10), gx_dps=30) for i in range(50)]
        self.assertFalse(mod.summarize(moving, 3)["accepted"])

    def test_ignores_short_gyro_spike_but_rejects_ground_idle(self):
        rows = [row(i / 10) for i in range(41)]
        rows[20]["gx_dps"] = 100
        self.assertEqual(len(mod.stable_segments(rows)), 1)
        idle = [row(i / 10, throttle=997) for i in range(41)]
        self.assertEqual(mod.stable_segments(idle), [])


if __name__ == "__main__":
    unittest.main()
