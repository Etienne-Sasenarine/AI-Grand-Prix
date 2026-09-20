"""The contract, checked against the simulator that lives beside it.

``tests/test_contract.py`` checks the contract against the *flight* repo, and
skips when that is not checked out. This file checks it against the simulator
in this repo, which is always present, so it can never skip.

Both directions matter and they fail differently. Drift against the flight repo
means the aircraft builds a vector the policy was not trained on. Drift within
this repo means the policy is not even trained on what the contract claims --
the hash stamped into the checkpoint becomes a lie, and the flight-side check
then passes while being wrong about everything.
"""

from __future__ import annotations

import pytest
import torch

from contract import camera, observation, plant
from dynamics import rate_control
from utils import aigp_obs


# --- plant ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, contract_value, sim_value",
    [
        ("THRUST_MIN", plant.THRUST_MIN, rate_control.THRUST_MIN),
        ("THRUST_HOVER", plant.THRUST_HOVER, rate_control.THRUST_HOVER),
        ("THRUST_MAX", plant.THRUST_MAX, rate_control.THRUST_MAX),
        ("RATE_LIMIT", plant.RATE_LIMIT, rate_control.RATE_LIMIT),
        ("G", plant.G, rate_control.G),
        ("MASS_KG", plant.MASS_KG, rate_control.DEFAULT_MASS_KG),
        ("PLANT_HOVER_RANGE", plant.PLANT_HOVER_RANGE, rate_control.PLANT_HOVER_RANGE),
        ("PLANT_TWR_RANGE", plant.PLANT_TWR_RANGE, rate_control.PLANT_TWR_RANGE),
        ("QUAD_SHARE", plant.MEASURED_QUAD_SHARE, rate_control.PLANT_QUAD_SHARE),
    ],
)
def test_plant_constant_matches_the_simulator(name, contract_value, sim_value):
    assert contract_value == sim_value, name


def test_zero_action_is_weight_under_the_contract_hover_point():
    """The single most important property of the action decode: a policy that
    outputs nothing should hover, not climb or sink."""
    stick, rates = rate_control.decode_aigp_action(torch.zeros(1, 4))
    assert float(stick) == pytest.approx(plant.THRUST_HOVER)
    assert torch.allclose(rates, torch.zeros(1, 3))

    newtons = rate_control.stick_to_newtons(stick, mass_kg=plant.MASS_KG)
    assert float(newtons) == pytest.approx(plant.MASS_KG * plant.G, rel=1e-5)


def test_action_rails_match_the_contract_ranges():
    lo, _ = rate_control.decode_aigp_action(torch.tensor([[-1.0, -1.0, -1.0, -1.0]]))
    hi, hi_rates = rate_control.decode_aigp_action(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    thrust_lo, thrust_hi = plant.ACTION_RANGES["thrust"]
    assert float(lo) == pytest.approx(thrust_lo)
    assert float(hi) == pytest.approx(thrust_hi)
    for i, name in enumerate(("roll_rate", "pitch_rate", "yaw_rate")):
        assert float(hi_rates[0, i]) == pytest.approx(plant.ACTION_RANGES[name][1]), name


def test_inertia_estimate_matches_the_event_that_writes_it():
    """The contract records the estimate; the env writes it into PhysX. If they
    part company the documented aircraft is not the simulated one."""
    import ast
    from pathlib import Path

    # Read the literal rather than importing: events.py needs Isaac Sim, and
    # this check is too important to skip on a laptop.
    source = (Path(__file__).resolve().parent.parent / "tasks" / "drone_racer"
              / "mdp" / "events.py").read_text()
    tree = ast.parse(source)
    written = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "BODY_INERTIA_DIAG" for t in node.targets
        ):
            written = ast.literal_eval(node.value)
    assert written is not None, "BODY_INERTIA_DIAG not found in events.py"
    assert tuple(written) == tuple(plant.INERTIA_DIAG)


def test_urdf_inertia_matches_the_estimate():
    """The URDF is not what loads -- the USD is -- but it is what a re-import
    would produce, so a stale one is a trap primed for whoever next rebuilds
    the asset. That is exactly how it came to hold a 0.5 kg aircraft's numbers
    against a 1.745 kg airframe."""
    import re
    from pathlib import Path

    urdf = (Path(__file__).resolve().parent.parent / "assets" / "5_in_drone"
            / "urdf" / "5_in_drone.urdf").read_text()
    match = re.search(r'<inertia ixx="([\d.]+)" iyy="([\d.]+)" izz="([\d.]+)"', urdf)
    assert match, "no inertia element in the URDF"
    assert tuple(float(g) for g in match.groups()) == pytest.approx(plant.INERTIA_DIAG)

    mass = re.search(r'<mass value="([\d.]+)"', urdf)
    assert mass and float(mass.group(1)) == pytest.approx(plant.MASS_KG)


# --- camera -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, contract_value, sim_value",
    [
        ("FRAME_W", camera.FRAME_W, aigp_obs.FRAME_W),
        ("FRAME_H", camera.FRAME_H, aigp_obs.FRAME_H),
        ("CAMERA_TILT_UP_DEG", camera.CAMERA_TILT_UP_DEG, aigp_obs.CAMERA_TILT_UP_DEG),
        ("GATE_OUTER_M", camera.GATE_OUTER_M, aigp_obs.GATE_OUTER_M),
        ("GATE_INNER_M", camera.GATE_INNER_M, aigp_obs.GATE_INNER_M),
        ("R2_DIST_MAX", camera.R2_DIST_MAX, aigp_obs.R2_DIST_MAX),
    ],
)
def test_camera_constant_matches_the_simulator(name, contract_value, sim_value):
    assert contract_value == sim_value, name


@pytest.mark.parametrize("axis", ["FX", "FY", "CX", "CY"])
def test_intrinsics_match_the_simulator(axis):
    assert getattr(camera, axis) == pytest.approx(getattr(aigp_obs, axis), abs=1e-6)


def test_distortion_matches_the_simulator():
    assert tuple(aigp_obs.DIST_K) == camera.DIST_K
    assert tuple(aigp_obs.DIST_P) == camera.DIST_P


def test_rendered_camera_matches_the_projected_keypoints():
    """The critic sees a rendered image; the policy sees projected corners. If
    the USD camera's focal length disagrees with the projection, the two
    describe different aircraft and only one of them is the one being flown.
    """
    import re
    from pathlib import Path

    cfg = (Path(__file__).resolve().parent.parent / "tasks" / "drone_racer"
           / "drone_racer_env_cfg.py").read_text()

    def _num(pattern):
        match = re.search(pattern, cfg)
        assert match, f"could not find {pattern} in the env config"
        return float(match.group(1))

    focal = _num(r"focal_length=([\d.]+)")
    h_aperture = _num(r"horizontal_aperture=([\d.]+)")
    v_aperture = _num(r"vertical_aperture=([\d.]+)")
    width = _num(r"width=(\d+)")
    height = _num(r"height=(\d+)")

    assert (width, height) == (camera.FRAME_W, camera.FRAME_H)
    assert width * focal / h_aperture == pytest.approx(camera.FX, abs=0.2)
    assert height * focal / v_aperture == pytest.approx(camera.FY, abs=0.2)


# --- observation ------------------------------------------------------------


def test_frame_width_matches_the_simulator():
    """v1 is corners + visibility + state + velocity + context, which is what
    the simulator calls FEATURE_DIM_VEL_CTX."""
    assert aigp_obs.FEATURE_DIM_VEL_CTX == observation.frame_dim("v1") == 51


def test_channel_names_match_the_simulator_in_order():
    """Order, not just width. A permuted vector has the right shape and the
    wrong meaning, and nothing downstream would notice."""
    assert tuple(aigp_obs.FEATURE_NAMES_VEL_CTX) == observation.frame_channels("v1")


@pytest.mark.parametrize(
    "name, contract_value, sim_value",
    [
        ("KEYPOINT_COUNT", observation.KEYPOINT_COUNT, aigp_obs.KEYPOINT_COUNT),
        ("NOT_SEEN", observation.NOT_SEEN, aigp_obs.NOT_SEEN),
        ("OFF_FRAME_MARGIN", observation.OFF_FRAME_MARGIN, aigp_obs.OFF_FRAME_MARGIN),
        ("GYRO_CLIP", observation.GYRO_CLIP, aigp_obs.GYRO_CLIP),
        ("VEL_CLIP", observation.VEL_CLIP, aigp_obs.VEL_CLIP),
    ],
)
def test_observation_constant_matches_the_simulator(name, contract_value, sim_value):
    assert contract_value == sim_value, name


# --- v2 / latency -----------------------------------------------------------


def _pack(actions=None):
    n = 2
    return aigp_obs.pack_observation(
        torch.rand(n, aigp_obs.KEYPOINT_COUNT, 2) * camera.FRAME_W,
        torch.ones(n, aigp_obs.KEYPOINT_COUNT, dtype=torch.bool),
        torch.zeros(n), torch.zeros(n), torch.zeros(n, 3),
        velocity_body_ned=torch.zeros(n, 3),
        gate_index=torch.zeros(n, dtype=torch.long),
        with_velocity=True, with_context=True, actions=actions,
    )


def test_simulator_v1_frame_is_the_width_the_contract_declares():
    assert _pack().shape[-1] == observation.frame_dim("v1")


def test_simulator_v2_frame_is_the_width_the_contract_declares():
    """The contract says 55 and 1760; if the simulator packs anything else the
    hash stamped into a checkpoint is describing a vector that does not exist.
    """
    actions = torch.zeros(2, len(observation.ACTION_CHANNELS))
    assert _pack(actions).shape[-1] == observation.frame_dim("v2")
    assert observation.observation_dim("v2") == observation.frame_dim("v2") * observation.HISTORY


def test_action_channel_count_matches_the_action_space():
    assert aigp_obs.ACTION_DIM == len(observation.ACTION_CHANNELS)
    assert aigp_obs.ACTION_DIM == len(plant.ACTION_NAMES)


def test_latency_defaults_are_on_and_action_history_is_available():
    """The combination that must never ship half-done. Modelling a delay the
    policy has no record of causing makes the plant worse, not better: the
    command it issued becomes an unexplainable disturbance."""
    import ast
    from pathlib import Path

    cfg_src = (Path(__file__).resolve().parent.parent / "tasks" / "drone_racer"
               / "drone_racer_env_cfg.py").read_text()
    assert "with_actions" in cfg_src, "v2 action channels are not wired into the env"
    assert "vision_delay_steps_range" in cfg_src, "vision delay is not wired in"

    # And the plant side is on by default.
    actions_src = (Path(__file__).resolve().parent.parent / "tasks" / "drone_racer"
                   / "mdp" / "actions.py").read_text()
    tree = ast.parse(actions_src)
    assert "ACTION_DELAY_STEPS_RANGE" in actions_src
    assert "RATE_TAU_S_RANGE" in actions_src
    del tree
