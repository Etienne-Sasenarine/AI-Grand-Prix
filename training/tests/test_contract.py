"""The contract, checked against itself and against the flight repo.

Two jobs. First, that the contract is internally coherent -- widths add up,
channels are unique, the hash is stable. Second, and the reason this file
earns its place, that it still matches what ``AI_GP`` actually flies.

The second set skips when the flight repo is not checked out beside this one,
so the suite stays runnable on a rented GPU box that only has the simulator.
That is a real gap rather than a clever trick: on a machine where these skip,
nothing is verifying that the vector the policy trains on is the vector the
aircraft will send. Run the full suite somewhere both repos exist before
trusting a checkpoint enough to fly it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from contract import camera, observation, plant, verify

# --- locating the flight repo ----------------------------------------------

from utils.flight_repo import find as _flight_repo  # noqa: E402


def _load_flight_module(name: str):
    """Import a module from the flight repo by path, without importing the repo."""
    repo = _flight_repo()
    if repo is None:
        pytest.skip("flight repo (AI_GP) not found beside this one")
    path = repo / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"{name}.py not present in the flight repo")
    spec = importlib.util.spec_from_file_location(f"_flight_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # numpy/config may be absent on a sim-only box
        pytest.skip(f"could not import {name} from the flight repo: {exc}")
    return module


# --- internal coherence -----------------------------------------------------


def test_v1_frame_is_fifty_one_channels():
    assert observation.frame_dim("v1") == 51


def test_v1_flattens_to_the_width_every_checkpoint_expects():
    assert observation.observation_dim("v1") == 1632
    assert observation.observation_dim("v1", lookahead=True) == 1639


def test_v2_adds_exactly_the_four_action_channels():
    v1 = set(observation.frame_channels("v1"))
    v2 = set(observation.frame_channels("v2"))
    assert v2 - v1 == set(observation.ACTION_CHANNELS)
    assert observation.observation_dim("v2") == 1760


def test_v2_is_a_strict_prefix_extension_of_v1():
    """Appending, not interleaving. A v1 reader must be able to read the first
    51 channels of a v2 frame and get exactly what it expects."""
    v1 = observation.frame_channels("v1")
    v2 = observation.frame_channels("v2")
    assert v2[: len(v1)] == v1


def test_channel_names_are_unique():
    for version in observation.VERSIONS:
        channels = observation.frame_channels(version)
        assert len(set(channels)) == len(channels), f"duplicate channel in {version}"


def test_segment_boundaries_agree_with_the_channel_list():
    channels = observation.frame_channels("v1")
    assert channels[observation.VISUAL_END - 1].startswith("vis")
    assert channels[observation.VISUAL_END] == "roll"
    assert channels[observation.STATE_END] == "vx"
    assert channels[observation.VEL_END] == "gate0"
    assert observation.CONTEXT_END == observation.frame_dim("v1")


def test_unknown_version_is_refused_by_name():
    with pytest.raises(ValueError, match="v3"):
        observation.frame_channels("v3")


def test_hash_is_stable_across_calls():
    assert verify.contract_hash() == verify.contract_hash()


def test_versions_hash_differently():
    assert verify.contract_hash("v1") != verify.contract_hash("v2")


def test_check_accepts_its_own_hash_and_rejects_others():
    verify.check(verify.contract_hash())
    with pytest.raises(verify.ContractMismatch, match="means something different"):
        verify.check("0" * 64)


def test_stamp_round_trips_through_check():
    stamp = verify.stamp("v1")
    verify.check(stamp["contract_hash"], stamp["contract_version"])


def test_randomised_plant_is_not_in_the_hash():
    """The policy is meant to see a spread of hover points and thrust curves.
    Hashing them would make every training run incompatible with every other."""
    fields = verify.contract_fields()
    blob = repr(fields)
    for excluded in ("plant_hover", "twr", "inertia", "mass"):
        assert excluded not in blob, f"{excluded} must not be part of the contract"


# --- camera -----------------------------------------------------------------


def test_intrinsics_at_native_resolution_are_the_calibration():
    fx, fy, cx, cy = camera.intrinsics_at(camera.CALIB_W, camera.CALIB_H)
    assert fx == pytest.approx(camera.CALIB_FX)
    assert fy == pytest.approx(camera.CALIB_FY)
    assert cx == pytest.approx(camera.CALIB_CX)
    assert cy == pytest.approx(camera.CALIB_CY)


def test_intrinsics_scale_per_axis_not_by_width_alone():
    """The bug this module exists to prevent. At 16:9 every convention agrees,
    which is why two different derivations coexisted unnoticed; off 16:9 they
    do not, and fy scaled by the width ratio is simply wrong."""
    _, fy_square, _, cy_square = camera.intrinsics_at(640, 480)
    assert fy_square == pytest.approx(camera.CALIB_FY * 480 / camera.CALIB_H)
    assert cy_square == pytest.approx(camera.CALIB_CY * 480 / camera.CALIB_H)
    # and that is not what scaling by width would have given
    assert fy_square != pytest.approx(camera.CALIB_FY * 640 / camera.CALIB_W)


def test_distortion_split_is_the_same_five_numbers_as_opencv_order():
    (k1, k2, k3), (p1, p2) = camera.distortion_split()
    assert camera.distortion_opencv() == (k1, k2, p1, p2, k3)


def test_field_of_view_matches_the_measured_lens():
    assert camera.HFOV_DEG == pytest.approx(72.8, abs=0.3)


# --- agreement with the flight repo -----------------------------------------


def test_matches_flight_race_obs_layout():
    race_obs = _load_flight_module("race_obs")
    assert tuple(race_obs.CORNER_CHANNELS) == observation.CORNER_CHANNELS
    assert tuple(race_obs.VIS_CHANNELS) == observation.VIS_CHANNELS
    assert tuple(race_obs.STATE_CHANNELS) == observation.STATE_CHANNELS
    assert tuple(race_obs.VEL_CHANNELS) == observation.VEL_CHANNELS
    assert tuple(race_obs.CONTEXT_CHANNELS) == observation.CONTEXT_CHANNELS


def test_matches_flight_race_obs_constants():
    race_obs = _load_flight_module("race_obs")
    assert race_obs.KEYPOINT_COUNT == observation.KEYPOINT_COUNT
    assert race_obs.NOT_SEEN == observation.NOT_SEEN
    assert race_obs.GYRO_CLIP == observation.GYRO_CLIP
    assert race_obs.VEL_CLIP == observation.VEL_CLIP
    assert race_obs.N_GATES == observation.N_GATES
    assert race_obs.OFF_FRAME_MARGIN == observation.OFF_FRAME_MARGIN
    assert race_obs.VISUAL_END == observation.VISUAL_END
    assert race_obs.STATE_END == observation.STATE_END


def test_matches_flight_frame_width():
    race_obs = _load_flight_module("race_obs")
    assert race_obs.feature_dim(with_context=True, with_velocity=True) == observation.frame_dim("v1")


def test_matches_flight_camera_model():
    cm = _load_flight_module("camera_model")
    assert (cm.WIDTH, cm.HEIGHT) == (int(camera.FRAME_W), int(camera.FRAME_H))
    assert cm.FX == pytest.approx(camera.FX)
    assert cm.FY == pytest.approx(camera.FY)
    assert cm.CX == pytest.approx(camera.CX)
    assert cm.CY == pytest.approx(camera.CY)
    assert cm.CAMERA_TILT_UP_DEG == camera.CAMERA_TILT_UP_DEG
    assert cm.GATE_INNER_M == camera.GATE_INNER_M
    assert tuple(float(d) for d in cm.D) == camera.distortion_opencv()


def test_matches_flight_action_names():
    race_obs = _load_flight_module("race_obs")
    assert tuple(race_obs.LABEL_NAMES) == plant.ACTION_NAMES


def test_rate_envelope_is_shared_with_the_flight_repo():
    """The rate rails are genuinely one contract: what +-1 means in rad/s has
    to be identical or the policy commands a different aircraft than it
    trained on."""
    race_obs = _load_flight_module("race_obs")
    for name in ("roll_rate", "pitch_rate", "yaw_rate"):
        assert race_obs.ACTION_RANGES[name] == pytest.approx(plant.ACTION_RANGES[name]), name


def test_imitation_thrust_bins_are_deliberately_narrower_than_the_plant_rail():
    """Not drift. ``race_obs.ACTION_RANGES`` discretises actions for behaviour
    cloning and bins thrust over the range a human actually flew, so the bins
    spend resolution on demonstrated stick positions. The PPO plant decodes to
    the full rail. Asserted so the relationship cannot silently invert -- a BC
    ceiling *above* the plant rail would encode actions the aircraft cannot be
    commanded to take."""
    race_obs = _load_flight_module("race_obs")
    assert race_obs.ACTION_RANGES["thrust"] == pytest.approx(plant.BC_THRUST_BIN_RANGE)
    bc_lo, bc_hi = plant.BC_THRUST_BIN_RANGE
    assert bc_lo == plant.THRUST_MIN
    assert bc_hi < plant.THRUST_MAX
