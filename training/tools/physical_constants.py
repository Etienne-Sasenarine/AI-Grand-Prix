"""Offline worksheet/calculator for PHYSICAL_TUNING.md (standard library only).

Run with --template measurements.json, fill measured fields, then:
    python tools/physical_constants.py measurements.json --output constants.json
No aircraft connection or automatic source edits. Null means not measured.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from pathlib import Path

G = 9.80665
AXES = ('roll', 'pitch', 'yaw')
TARGETS = {
    'mass_kg': 'dynamics/rate_control.py DEFAULT_MASS_KG; tasks/drone_racer/mdp/actions.py ControlActionCfg.mass_kg; assets/5_in_drone/urdf/5_in_drone.urdf mass',
    'hover_thrust': 'dynamics/rate_control.py THRUST_HOVER; ControlActionCfg.hover_thrust; AI_GP/config.py HOVER_THRUST',
    'thrust_min': 'dynamics/rate_control.py THRUST_MIN; ControlActionCfg.min_thrust; AI_GP/config.py MIN_THRUST',
    'thrust_max': 'dynamics/rate_control.py THRUST_MAX; ControlActionCfg.max_thrust; AI_GP/config.py MAX_THRUST',
    'rate_limit': 'dynamics/rate_control.py RATE_LIMIT; ControlActionCfg.rate_limit; AI_GP/race_obs.py ACTION_RANGES; verify FC limits separately',
    'rate_kp': 'tasks/drone_racer/mdp/actions.py ControlActionCfg.rate_kp (roll, pitch, yaw)',
    'rate_kd': 'tasks/drone_racer/mdp/actions.py ControlActionCfg.rate_kd (roll, pitch, yaw)',
    'moment_limit': 'tasks/drone_racer/mdp/actions.py ControlActionCfg.moment_limit (N m)',
    'inertia': 'assets/5_in_drone/urdf/5_in_drone.urdf ixx, iyy, izz (kg m^2)',
    'drag': 'utils/aigp_obs.py DEFAULT_DRAG_BODY; AI_GP/config.py DRAG_KX/Y/Z (FRD, 1/s)',
    'camera': 'utils/aigp_obs.py and AI_GP/camera_model.py FX/FY/CX/CY/CAMERA_TILT_UP_DEG; review image size consumers if resolution changes',
    'rate_signs': 'AI_GP/config.py RATE_SIGN_ROLL/PITCH/YAW only',
    'timing': 'Log only; do not change sim.dt/decimation automatically',
    'imu': 'Diagnostic only; do not add accelerometer to policy observations',
}


def template():
    """Deliberately no invented measured values, including camera intrinsics."""
    return {
        'mass_kg': None,
        'thrust_min': None, 'thrust_max': None,
        'hover_trials': [],
        'rate_sign_trials': {axis: None for axis in AXES},
        'rate_steps': [],
        'inertia': {axis: None for axis in AXES},
        'drag_samples': [],
        'camera': {'width': None, 'height': None, 'fx': None, 'fy': None,
                   'cx': None, 'cy': None, 'same_height_gate_cy': None},
        'timing': {'control_t_s': [], 'camera_t_s': [], 'command_to_gyro_ms': []},
        'imu_rest': [],
    }


def number(value, label, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label}: expected a finite number')
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f'{label}: outside [{low}, {high}]')
    return float(value)


def positive(value, label):
    value = number(value, label, 0)
    if value == 0:
        raise ValueError(f'{label}: must be positive')
    return value


def vector(value, label):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f'{label}: expected three FRD components')
    return [number(x, label) for x in value]


def generate(data):
    unknown = set(data) - set(template())
    if unknown:
        raise ValueError(f'Unknown input fields: {sorted(unknown)}')
    out = {'constants': {}, 'diagnostics': {}, 'notes': [], 'targets': TARGETS}
    c, d, notes = out['constants'], out['diagnostics'], out['notes']
    for key in ('mass_kg', 'thrust_min', 'thrust_max'):
        if data.get(key) is not None:
            c[key] = positive(data[key], key) if key == 'mass_kg' else number(data[key], key, 0, 1)
    if c.get('thrust_min', 0) >= c.get('thrust_max', 1):
        raise ValueError('thrust_min must be below thrust_max')

    # Each trial is a verified level, constant-altitude 3-5 second window.
    hover = []
    for trial in data.get('hover_trials', []):
        duration = positive(trial['duration_s'], 'hover duration_s')
        tilt = abs(number(trial['max_abs_tilt_deg'], 'hover tilt', 0))
        rate = number(trial['max_abs_rate_rad_s'], 'hover rate', 0)
        climb = abs(number(trial['climb_m_s'], 'hover climb'))
        stick = number(trial['mean_stick'], 'hover mean_stick', 0.000001, 1)
        if duration >= 3 and tilt < 8 and rate < math.radians(20) and climb <= 0.08:
            hover.append(stick)
    d['accepted_hover_trials'] = len(hover)
    if len(hover) >= 3:
        c['hover_thrust'] = stats.median(hover)
        if not c.get('thrust_min', 0) < c['hover_thrust'] < c.get('thrust_max', 1):
            raise ValueError('hover must lie strictly inside thrust rails')
        if 'mass_kg' in c:
            d['hover_newtons'] = c['mass_kg'] * G
            d['newtons_per_stick'] = c['mass_kg'] * G / c['hover_thrust']
    else:
        notes.append('Hover pending: need at least three accepted level, constant-altitude trials.')

    signs = {}
    for axis, trial in data.get('rate_sign_trials', {}).items():
        if axis not in AXES:
            raise ValueError(f'Unknown sign axis: {axis}')
        if trial is not None:
            current = number(trial['current_sign'], 'current_sign', -1, 1)
            if current not in (-1, 1):
                raise ValueError('current_sign must be -1 or +1')
            cmd = number(trial['command_rad_s'], 'sign command')
            response = number(trial['gyro_delta_rad_s'], 'sign gyro delta')
            if abs(cmd) < 0.05 or abs(response) < 0.05:
                notes.append(f'{axis} sign pending: insufficient command/response.')
            else:
                signs[axis] = int(current * (1 if cmd * response > 0 else -1))
    if signs:
        c['rate_signs'] = signs

    inertia = {}
    for axis, measurement in data.get('inertia', {}).items():
        if axis not in AXES:
            raise ValueError(f'Unknown inertia axis: {axis}')
        if measurement is None:
            continue
        method = measurement['method']
        if method == 'direct':
            inertia[axis] = positive(measurement['kg_m2'], 'inertia')
        elif method == 'known_torque':
            torque = number(measurement['net_torque_nm'], 'net torque')
            alpha = number(measurement['alpha_rad_s2'], 'angular acceleration')
            if alpha == 0 or torque * alpha <= 0:
                raise ValueError('known torque and acceleration must be nonzero and have matching signs')
            inertia[axis] = torque / alpha
        elif method == 'bifilar':
            mass = positive(measurement['suspended_mass_kg'], 'suspended mass')
            spacing = positive(measurement['string_spacing_m'], 'string spacing')
            length = positive(measurement['string_length_m'], 'string length')
            period = positive(measurement['period_s'], 'period')
            fixture = number(measurement.get('fixture_inertia_kg_m2', 0), 'fixture inertia', 0)
            inertia[axis] = positive(mass * G * (spacing / 2)**2 * period**2 /
                                     (4 * math.pi**2 * length) - fixture, 'drone inertia')
        else:
            raise ValueError(f'Unknown inertia method: {method}')
    if inertia:
        c['inertia'] = inertia

    steps = {axis: [] for axis in AXES}
    for step in data.get('rate_steps', []):
        axis = step['axis']
        if axis not in steps:
            raise ValueError(f'Unknown rate axis: {axis}')
        command = number(step['command_rad_s'], 'step command')
        if abs(command) < 0.1:
            raise ValueError('Rate step command must have magnitude >= 0.1 rad/s')
        steady = number(step['steady_rad_s'], 'step steady')
        peak = number(step['peak_rad_s'], 'step peak')
        steps[axis].append((step, abs(command), steady * math.copysign(1, command), abs(peak)))
    rails, gains, damping, moments = {}, {}, {}, {}
    for axis, trials in steps.items():
        rail_trials = [s for s in trials if s[1] >= 3.2 and s[0].get('settled') is True]
        if rail_trials:
            steady = min(s[2] for s in rail_trials)
            if steady > 0:
                rails[axis] = 3.2 if steady >= 3.0 else steady
        fits = []
        for step, command, steady, peak in trials:
            if step.get('settled') is not True or step.get('unsaturated') is not True:
                continue
            if step.get('rise_10_90_final_s') is None or axis not in inertia:
                continue
            rise = positive(step['rise_10_90_final_s'], 'rise time')
            gain = steady / command
            if not 0 < gain <= 1 or peak > steady * 1.05:
                notes.append(f'{axis}: ringing/overshoot or gain outside (0,1]; first-order PD fit omitted.')
                continue
            # I*w_dot = kp*u - (kp+kd)*w. Rise is relative to FINAL response.
            total = inertia[axis] * math.log(9) / rise
            fits.append((gain * total, (1 - gain) * total))
        if fits:
            gains[axis] = stats.median(f[0] for f in fits)
            damping[axis] = stats.median(f[1] for f in fits)
        saturated = [positive(s[0]['saturated_alpha_rad_s2'], 'saturated alpha')
                     for s in trials if s[0].get('saturation_confirmed') is True
                     and s[0].get('saturated_alpha_rad_s2') is not None]
        if saturated and axis in inertia:
            moments[axis] = inertia[axis] * stats.median(saturated)
    d['rate_rails_by_axis'] = rails
    if len(rails) == 3:
        c['rate_limit'] = min(rails.values())
        if abs(c['rate_limit'] - 3.2) > 0.3:
            notes.append('Rate rail changed >0.3 rad/s: warm-start review required.')
    else:
        notes.append('Rate limit pending: settled rail tests at >=3.2 rad/s required on all axes; a slow step alone does not prove clipping.')
    for key, values in [('rate_kp', gains), ('rate_kd', damping), ('moment_limit', moments)]:
        if values:
            c[key] = values

    # Input acceleration is the time derivative of independent FRD velocity,
    # NOT accelerometer specific force. Remove gravity, thrust and transport.
    drag = {axis: [] for axis in ('x', 'y', 'z')}
    for sample in data.get('drag_samples', []):
        v = vector(sample['v_body_m_s'], 'velocity')
        a = vector(sample['dv_body_dt_m_s2'], 'velocity derivative')
        g = vector(sample['gravity_body_m_s2'], 'body gravity')
        w = vector(sample['omega_body_rad_s'], 'body rate')
        tm = number(sample['thrust_accel_m_s2'], 'T/m', 0)
        cross = [w[1]*v[2]-w[2]*v[1], w[2]*v[0]-w[0]*v[2], w[0]*v[1]-w[1]*v[0]]
        for i, axis in enumerate(drag):
            if abs(v[i]) > 0.2:
                drag[axis].append((v[i], a[i] - g[i] + (tm if i == 2 else 0) + cross[i]))
    fitted = {}
    d['drag'] = {}
    for i, (axis, samples) in enumerate(drag.items()):
        if len(samples) < 20:
            continue
        k = sum(v*a for v, a in samples) / sum(v*v for v, _ in samples)
        d['drag'][axis] = {'samples': len(samples), 'rmse_m_s2': math.sqrt(stats.mean((a-k*v)**2 for v, a in samples))}
        if k < 0:
            fitted[axis] = k
            if abs(abs(k) / (0.15 if i == 2 else 0.5) - 1) > 0.3:
                notes.append(f'Drag {axis} changed >30% from guide baseline: warm-start review required.')
        else:
            notes.append(f'Drag {axis}: nonnegative fit rejected; check frames, thrust and velocity truth.')
    if fitted:
        c['drag'] = fitted

    cam = data.get('camera', {})
    if all(cam.get(k) is not None for k in ('width', 'height', 'fx', 'fy', 'cx', 'cy', 'same_height_gate_cy')):
        camera = {k: positive(cam[k], k) for k in ('width', 'height', 'fx', 'fy')}
        for k in ('width', 'height'):
            if not camera[k].is_integer():
                raise ValueError(f'{k} must be an integer')
        camera.update({k: number(cam[k], k) for k in ('cx', 'cy')})
        camera['tilt_up_deg'] = math.degrees(math.atan((number(cam['same_height_gate_cy'], 'gate cy') - camera['cy']) / camera['fy']))
        c['camera'] = camera
        notes.append('Camera estimate assumes level airframe, same-height gate center, pinhole image and negligible roll. Changed intrinsics/tilt require retraining review.')

    timing = {}
    for key in ('control_t_s', 'camera_t_s'):
        times = [number(t, key) for t in data.get('timing', {}).get(key, [])]
        if len(times) >= 2:
            intervals = [b-a for a, b in zip(times, times[1:])]
            if min(intervals) <= 0:
                raise ValueError(f'{key} must be strictly increasing')
            timing[key] = {'mean_hz': 1 / stats.mean(intervals), 'median_period_ms': 1000 * stats.median(intervals), 'max_period_ms': 1000 * max(intervals)}
    delays = [number(x, 'delay ms', 0) for x in data.get('timing', {}).get('command_to_gyro_ms', [])]
    if delays:
        timing['median_delay_ms'] = stats.median(delays)
        if timing['median_delay_ms'] > 30:
            notes.append('Delay >30 ms: if keeper overshoots, evaluate action delay in one warm-start.')
    if timing:
        c['timing'] = timing

    rest = data.get('imu_rest', [])
    if rest:
        ts = [number(s['t_s'], 'IMU timestamp') for s in rest]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError('IMU timestamps must be strictly increasing')
        gyro = [vector(s['gyro_rad_s'], 'gyro') for s in rest]
        accel = [vector(s['accel_m_s2'], 'accel') for s in rest]
        if ts[-1] - ts[0] >= 5:
            bias = [stats.mean(s[i] for s in gyro) for i in range(3)]
            c['imu'] = {'gyro_bias_rad_s': bias, 'mean_accel_m_s2': [stats.mean(s[i] for s in accel) for i in range(3)], 'mean_accel_norm_m_s2': stats.mean(math.sqrt(sum(x*x for x in s)) for s in accel)}
            if max(map(abs, bias)) > 0.05:
                notes.append('Rest gyro bias >0.05 rad/s: FC zeroing needs review.')
        else:
            notes.append('IMU pending: need at least 5 seconds stationary data.')
    out['pending'] = [key for key in TARGETS if key not in c or
                      isinstance(c[key], dict) and key in ('rate_signs', 'inertia', 'rate_kp', 'rate_kd', 'moment_limit', 'drag') and len(c[key]) < 3]
    notes.append('PD/moment estimates assume isolated principal-axis steps and negligible gyroscopic coupling. Verify at Isaac 120 Hz; inertia and gains cannot be independently inferred from rise time alone.')
    return out


def report(result):
    lines = ['# Physical constants — measured candidates', '', 'Review before copying. Missing values remain pending; no defaults are presented as measurements.', '']
    for key, value in result['constants'].items():
        lines.extend([f'## {key}', '', f'`{json.dumps(value, sort_keys=True)}`', '', TARGETS[key], ''])
    lines.extend(['## Pending', '', ', '.join(result['pending']) or 'None', '', '## Notes', ''])
    lines.extend(f'- {note}' for note in result['notes'])
    lines.extend(['', '## Diagnostics', '', '```json', json.dumps(result['diagnostics'], indent=2), '```', ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path, nargs='?')
    parser.add_argument('--template', type=Path, help='write an empty measurement worksheet (refuses overwrite)')
    parser.add_argument('--output', type=Path, help='write JSON and a companion Markdown report')
    args = parser.parse_args(argv)
    try:
        if args.template:
            if args.measurements or args.output:
                parser.error('--template cannot be combined with measurements/--output')
            with args.template.open('x', encoding='utf-8') as f:
                f.write(json.dumps(template(), indent=2) + '\n')
            print(f'Created {args.template}')
            return 0
        if not args.measurements:
            parser.error('provide measurements.json or --template PATH')
        result = generate(json.loads(args.measurements.read_text(encoding='utf-8')))
        if args.output:
            if args.output.suffix.lower() != '.json':
                parser.error('--output must end in .json')
            paths = (args.output, args.output.with_suffix('.md'))
            if any(p.exists() for p in paths):
                raise ValueError('Output already exists; choose a new output name')
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            paths[1].write_text(report(result), encoding='utf-8')
        print(report(result))
        return 0
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
