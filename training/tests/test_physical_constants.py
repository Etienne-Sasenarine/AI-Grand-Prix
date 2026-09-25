"""Offline physical identification regression tests; no Isaac dependency."""
import importlib.util
import math
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('physical_constants', Path(__file__).resolve().parents[1] / 'tools' / 'physical_constants.py')
pc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pc)


class PhysicalConstantsTests(unittest.TestCase):
    def test_empty_has_no_fabricated_constants(self):
        result = pc.generate(pc.template())
        self.assertEqual(result['constants'], {})
        self.assertEqual(set(result['pending']), set(pc.TARGETS))

    def test_hover_rejects_motion_uses_three_trial_median(self):
        trial = dict(duration_s=4, max_abs_tilt_deg=2, max_abs_rate_rad_s=0.1, climb_m_s=0)
        data = {'mass_kg': 0.62, 'hover_trials': [dict(trial, mean_stick=x) for x in (0.25, 0.26, 0.27)]}
        data['hover_trials'].append(dict(trial, mean_stick=0.9, climb_m_s=1))
        result = pc.generate(data)
        self.assertEqual(result['constants']['hover_thrust'], 0.26)
        self.assertAlmostEqual(result['diagnostics']['hover_newtons'], 0.62*pc.G)
        data['hover_trials'] = data['hover_trials'][:2]
        self.assertNotIn('hover_thrust', pc.generate(data)['constants'])

    def test_known_first_order_plant_recovery(self):
        inertia, kp, kd = 0.003, 0.08, 0.003
        gain = kp/(kp+kd)
        step = dict(axis='roll', command_rad_s=2, steady_rad_s=2*gain, peak_rad_s=2*gain,
                    settled=True, unsaturated=True, rise_10_90_final_s=math.log(9)*inertia/(kp+kd))
        result = pc.generate({'inertia': {'roll': {'method': 'direct', 'kg_m2': inertia}}, 'rate_steps': [step]})
        self.assertAlmostEqual(result['constants']['rate_kp']['roll'], kp)
        self.assertAlmostEqual(result['constants']['rate_kd']['roll'], kd)
        self.assertNotIn('rate_limit', result['constants'])
        step['peak_rad_s'] = 2.5
        self.assertNotIn('rate_kp', pc.generate({'inertia': {'roll': {'method': 'direct', 'kg_m2': inertia}}, 'rate_steps': [step]})['constants'])

    def test_rails_require_all_axes_and_use_steady_not_peak(self):
        steps = [dict(axis=a, command_rad_s=3.2, steady_rad_s=2.6, peak_rad_s=4, settled=True) for a in pc.AXES]
        self.assertNotIn('rate_limit', pc.generate({'rate_steps': steps[:2]})['constants'])
        self.assertEqual(pc.generate({'rate_steps': steps})['constants']['rate_limit'], 2.6)

    def test_drag_removes_gravity_thrust_and_rotating_frame(self):
        samples = []
        expected = [-0.5, -0.4, -0.15]
        for n in range(25):
            v = [1+n/25, 0.8, -0.5]
            w = [0.1, -0.2, 0.3]
            g = [1, 2, 9.5]
            cross = [w[1]*v[2]-w[2]*v[1], w[2]*v[0]-w[0]*v[2], w[0]*v[1]-w[1]*v[0]]
            a = [g[i] - (10 if i == 2 else 0) + expected[i]*v[i] - cross[i] for i in range(3)]
            samples.append(dict(v_body_m_s=v, dv_body_dt_m_s2=a, gravity_body_m_s2=g, omega_body_rad_s=w, thrust_accel_m_s2=10))
        result = pc.generate({'drag_samples': samples})
        for axis, k in zip('xyz', expected):
            self.assertAlmostEqual(result['constants']['drag'][axis], k)

    def test_bifilar_and_moment(self):
        mass, spacing, length, inertia = 0.62, 0.2, 1, 0.003
        period = math.sqrt(inertia * 4*math.pi**2*length/(mass*pc.G*(spacing/2)**2))
        data = {'inertia': {'yaw': dict(method='bifilar', suspended_mass_kg=mass, string_spacing_m=spacing, string_length_m=length, period_s=period)},
                'rate_steps': [dict(axis='yaw', command_rad_s=-2, steady_rad_s=-1.9, peak_rad_s=-2, saturation_confirmed=True, saturated_alpha_rad_s2=100)]}
        self.assertAlmostEqual(pc.generate(data)['constants']['moment_limit']['yaw'], 0.3)

    def test_camera_sign_timing_and_imu(self):
        data = {'camera': dict(width=640, height=360, fx=423.6, fy=422.3, cx=320, cy=180, same_height_gate_cy=180+422.3*math.tan(math.radians(20))),
                'rate_sign_trials': {'pitch': dict(current_sign=-1, command_rad_s=0.2, gyro_delta_rad_s=-0.18)},
                'timing': {'control_t_s': [0, 0.01, 0.02]},
                'imu_rest': [dict(t_s=i*0.01, gyro_rad_s=[0.01, 0, 0], accel_m_s2=[0, 0, -pc.G]) for i in range(501)]}
        c = pc.generate(data)['constants']
        self.assertAlmostEqual(c['camera']['tilt_up_deg'], 20)
        self.assertEqual(c['rate_signs']['pitch'], 1)
        self.assertAlmostEqual(c['timing']['control_t_s']['mean_hz'], 100)
        self.assertAlmostEqual(c['imu']['gyro_bias_rad_s'][0], 0.01)

    def test_invalid_measurements_rejected(self):
        for data in ({'mass_kg': -1}, {'mass_kg': float('nan')}, {'thrust_min': 0.9, 'thrust_max': 0.1},
                     {'timing': {'control_t_s': [1, 1]}}, {'mass_typo': 1}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                pc.generate(data)


if __name__ == '__main__':
    unittest.main()
