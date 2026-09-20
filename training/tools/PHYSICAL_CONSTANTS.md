# Generate physical constants offline

From `AI_GP/isaac_drone_racer`:

```bash
python tools/physical_constants.py --template measurements.json
# Fill measurements.json using the schemas below, then:
python tools/physical_constants.py measurements.json --output constants.json
```

This standard-library script writes candidate constants and diagnostics to JSON
and a companion Markdown report, including the source locations to update.
It never connects to MAVLink, arms, changes configuration, or trains. Existing
output files are protected; choose a new filename for each measurement session.
Nulls and empty lists mean unmeasured. Partial axis results stay marked pending.
The numbers below illustrate the input format, **not measured aircraft values**.

Use the acquisition procedure in `../PHYSICAL_TUNING.md`. The existing
`AI_GP/tools/pad_constants.py` can acquire hover/rate logs; this calculator takes
validated measurement summaries, not those raw CSV files. Raw drag and IMU
samples can be supplied as JSON arrays. All vectors are body FRD; rates are
rad/s, times seconds, distances meters, mass kg.

## Phase 0: signs

For each `rate_sign_trials` axis, replace null with:

```json
{"current_sign": -1, "command_rad_s": 0.2, "gyro_delta_rad_s": -0.18}
```

Command means requested rate **before** the current wire sign; gyro delta is
response minus baseline in the desired NED/FRD convention. The output is the
absolute replacement sign (`current_sign` times response polarity). Hand tipping
alone checks sensor convention; it cannot identify command polarity.

## Phases 1–2: scale, hover and rails

Set `mass_kg` to the race-ready scale reading and `thrust_min`/`thrust_max` to
verified stick limits (0–1). Add at least three trials to `hover_trials`:

```json
{"mean_stick": 0.26, "duration_s": 4, "max_abs_tilt_deg": 3,
 "max_abs_rate_rad_s": 0.1, "climb_m_s": 0.01}
```

Only trials lasting >=3 s, tilt <8 degrees, rate <20 degrees/s and absolute
climb <=0.08 m/s qualify. The median of accepted trial means becomes hover.
Thrust conversion is `N = stick / hover * mass * 9.80665`.

## Phases 3–4: rate envelope and plant

Add steps for each axis to `rate_steps`:

```json
{"axis": "roll", "command_rad_s": 2.0, "steady_rad_s": 1.92,
 "peak_rad_s": 1.94, "settled": true, "unsaturated": true,
 "rise_10_90_final_s": 0.08}
```

Measure rise between 10% and 90% of the **final response**, starting from zero
rate, not 10–90% of command (the existing pad CSV helper uses command). Mark
settled only after inspecting a stable tail. A 0.4 s pulse may be too short.
Use positive and negative steps where practical. Envelope generation requires
settled >=3.2 rad/s magnitude tests for all three axes. It conservatively uses
the lowest steady response; overshoot peaks are not treated as sustainable rails.
Inspect low steady tracking for clipping versus controller steady-state error
before adopting a lower limit.

Supply independently measured `inertia` per axis (`roll=ixx`, `pitch=iyy`,
`yaw=izz`) using one of:

```json
{"method": "direct", "kg_m2": 0.003}
```

```json
{"method": "known_torque", "net_torque_nm": 0.03, "alpha_rad_s2": 10}
```

```json
{"method": "bifilar", "suspended_mass_kg": 0.62, "string_spacing_m": 0.20,
 "string_length_m": 1.0, "period_s": 1.4, "fixture_inertia_kg_m2": 0}
```

Bifilar assumes parallel equal-length strings, equal attachment spacing above
and below, small oscillations, and suspension about the measured principal axis.
Use total suspended mass, subtract independently known fixture inertia.

For unsaturated non-ringing steps, the implemented model is
`I*w_dot = kp*u - (kp+kd)*w`. With steady gain `K=w_final/u` and
`tau=t10_90/log(9)`, `kp=I*K/tau`, `kd=I*(1-K)/tau`.
Increasing this implementation's `kd` reduces both time constant and steady gain;
it is rate damping, not a derivative-of-error term. Overshoot >5% or gain
outside (0,1] suppresses the fit. A first-order fit cannot reproduce FC ringing
or transport delay. It also does not validate stability at the 120 Hz timestep.

For a separate step with a verified torque-saturated, nearly constant slope,
add `"saturation_confirmed": true, "saturated_alpha_rad_s2": 100`.
Use the positive acceleration magnitude. The moment estimate is `I*alpha`.
No moment is inferred from an ordinary unsaturated rise. Check fitted PD,
moments and independently measured inertia together in Isaac before use.

## Phase 5: drag

Add at least 20 valid, independent-velocity samples per excited axis:

```json
{"v_body_m_s": [1, 0, 0], "dv_body_dt_m_s2": [-0.5, 0, 0],
 "gravity_body_m_s2": [0, 0, 9.80665], "omega_body_rad_s": [0, 0, 0],
 "thrust_accel_m_s2": 9.80665}
```

`dv_body_dt_m_s2` is the derivative of velocity expressed in the rotating body
frame, **not IMU specific force**. Obtain velocity from independent optical flow,
MoCap or suitable vision, smooth/differentiate it, and align all timestamps.
`thrust_accel_m_s2 = stick/hover*g`.
The regression removes gravity, thrust and `-omega cross v` before fitting
`k = sum(v*residual)/sum(v*v)` on `abs(v)>0.2`. Positive/zero fits are rejected.
The report includes residual RMSE; sample count alone does not validate a fit.
Unexcited axes stay pending; keep existing values until measured.

## Phase 6: camera

Fill all `camera` fields from calibration. Width/height are image dimensions,
fx/fy focal lengths in pixels, cx/cy the calibrated principal point.
`same_height_gate_cy` is the measured image y of the gate center with level
airframe and gate center at camera height. Tilt is
`atan((same_height_gate_cy-cy)/fy)`. This single observation cannot independently
identify focal length and tilt; calibrate intrinsics first.

The guide's 320-pixel focal lengths are historical: this branch currently has
FX=423.6, FY=422.3 in both camera implementations. At 20 degrees, CY=180,
the expected same-height gate center is approximately 333.7 px. Do not blindly
copy the old 296 px criterion from the venue helper.

## Phases 7–8: timing and IMU diagnostics

`timing.control_t_s` and `camera_t_s` are strictly increasing timestamp arrays.
`command_to_gyro_ms` contains independently measured paired response delays;
the calculator summarizes these, it does not estimate synchronization or delay.

Add >=5 s of uniformly sampled stationary IMU data to `imu_rest`:

```json
{"t_s": 0, "gyro_rad_s": [0.01, -0.02, 0.005],
 "accel_m_s2": [0, 0, -9.80665]}
```

Gyro bias is the per-axis mean. Accel mean/norm are diagnostic. This is not an
Allan-variance noise model or a change to policy observations. No inferred
timing/IMU values are automatically written into the simulator.
