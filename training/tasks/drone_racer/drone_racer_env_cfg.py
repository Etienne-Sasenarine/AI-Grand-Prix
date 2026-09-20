# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .track_generator import generate_track

from assets.five_in_drone import FIVE_IN_DRONE  # isort:skip


@configclass
class DroneRacerSceneCfg(InteractiveSceneCfg):

    # ground plane
    #
    # The collision plane is infinite either way, but the default grid mesh is
    # only 100 m square and centred on the origin. Centre it on the PQ hall
    # (20.9 × 50.3 m course inside a 60 × 21 m footprint).
    # Dark hangar floor (VQ2 is nearly black with faint markings). The default
    # Isaac grid reads as a bright lab and fights the gate emission.
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(120.0, 120.0),
            color=(0.04, 0.04, 0.05),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(10.5, 25.0, 0.0)),
    )

    # track
    #
    # Physical Qualifier map from the organizer gate-coordinate table
    # (Drone_Race_Track_Gate_Coordinates_with_doublegate_5800.pdf): X/Y in ft from
    # the top-left of the 85 x 165 ft boundary, rotation clockwise from north.
    # Here x = X_ft * 0.3048 (east), y = (165 - Y_ft) * 0.3048 (north).
    # The previous hand-digitised track was a 0.935x copy of this layout
    # (similarity fit, residual <= 0.42 m); G5 and G8 headings were 35 and 13 deg off.
    # Heights: 2.7 m frames stand on the floor -> opening centre 1.35 m; the double
    # gate is two stacked frames -> upper opening 4.05 m. Confirm on site.
    # Isaac order is unchanged (index 0 = official gate 6) so the one-hot context
    # of existing checkpoints keeps its meaning.
    track: RigidObjectCollectionCfg = generate_track(
        track_config={
            # Centre gate, flown east (split-S: approached from the west)
            "1": {  # official G6
                "pos": (12.50, 24.08, 1.35),
                "yaw": 0.0000,
            },
            "2": {  # official G7
                "pos": (21.49, 17.80, 1.35),
                "yaw": -1.5708,
            },
            "3": {  # official G8
                "pos": (18.29, 10.97, 1.35),
                "yaw": -2.1817,
            },
            # Stacked double gate: upper opening southbound
            "4": {  # official G9 top
                "pos": (12.10, 5.27, 4.05),
                "yaw": -1.5708,
            },
            # then lower opening northbound
            "5": {  # official G9 bottom
                "pos": (12.10, 5.27, 1.35),
                "yaw": 1.5708,
            },
            "6": {  # official G10
                "pos": (3.96, 16.76, 1.35),
                "yaw": 1.5708,
            },
            # Race start: official gate 1 (Play fixed_start_idx = 6)
            "7": {  # official G1
                "pos": (3.66, 28.65, 1.35),
                "yaw": 1.5708,
            },
            "8": {  # official G2
                "pos": (4.88, 38.40, 1.35),
                "yaw": 1.5708,
            },
            "9": {  # official G3
                "pos": (12.19, 45.11, 1.35),
                "yaw": 0.0000,
            },
            "10": {  # official G4
                "pos": (21.95, 38.10, 1.35),
                "yaw": -1.5708,
            },
            "11": {  # official G5
                "pos": (20.12, 28.96, 1.35),
                "yaw": -2.1817,
            },
        }
    )

    # robot
    robot: ArticulationCfg = FIVE_IN_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    collision_sensor: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", debug_vis=True)
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/body", debug_vis=False)
    # On-site ChArUco (1920x1080): fx=1302.94 fy=1303.11 cx=952.29 cy=529.04.
    # Policy frame is 640x360 so fx≈434.3. Isaac pinhole:
    #   fx = width  * focal_length / horizontal_aperture = 640 * 12 / 17.684
    #   fy = height * focal_length / vertical_aperture   = 360 * 12 / 9.946
    # Distortion is applied on the keypoint projector, not this render camera.
    # The camera shares the body origin and is pitched 20 deg up; the quaternion
    # is a -20 deg rotation about body +Y (left), i.e. optical axis
    # cos(20)*forward + sin(20)*up.
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.9848078, 0.0, -0.1736482, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            horizontal_aperture=17.684,
            vertical_aperture=9.946,
            clipping_range=(0.05, 1.0e5),
        ),
        width=640,
        height=360,
    )

    # lights
    # Bright hangar: cool-white ceiling grid does the work. The dome is only
    # fill so unlit corners aren't black; hide it in the primary ray so the
    # sky still reads as a ceiling, not an outdoor dome.
    # Override with DOME_INTENSITY / OVERHEAD_INTENSITY.
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.55, 0.62, 0.75),
            intensity=float(os.environ.get("DOME_INTENSITY", 280.0)),
            visible_in_primary_ray=False,
        ),
    )

    def __post_init__(self) -> None:
        # Cool-white disks over the 21 × 50 m PQ box. DiskLight emits
        # along local -Z, so identity orientation points at the floor.
        # Tune with OVERHEAD_INTENSITY; set OVERHEAD_LIGHTS=0 to disable.
        if not int(os.environ.get("OVERHEAD_LIGHTS", "1")):
            return
        intensity = float(os.environ.get("OVERHEAD_INTENSITY", 80000.0))
        z = float(os.environ.get("OVERHEAD_HEIGHT", 14.0))
        xs = [3.5, 10.5, 17.5]
        ys = [8.0, 25.0, 42.0]
        idx = 0
        for x in xs:
            for y in ys:
                setattr(
                    self,
                    f"overhead_{idx}",
                    AssetBaseCfg(
                        prim_path=f"/World/Overhead_{idx}",
                        spawn=sim_utils.DiskLightCfg(
                            radius=1.4,
                            color=(0.78, 0.86, 1.0),
                            intensity=intensity,
                            enable_color_temperature=True,
                            color_temperature=7800.0,
                            normalize=True,
                        ),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, z)),
                    ),
                )
                idx += 1


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(use_motor_model=False)


# Observation contract version. v1 is the 51-channel frame every checkpoint
# through pq_speed_best trained against; v2 appends the four action channels,
# making the frame 55 wide and the flattened vector 1760. Changing this
# invalidates checkpoints, which is why it is an explicit switch rather than a
# silent default.
_OBS_VERSION = os.environ.get("OBS_VERSION", "v2").strip().lower()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.

        Uses the AI Grand Prix ``race_obs`` state vector (projected gate
        keypoints + IMU + course context) instead of privileged full state.
        """

        aigp_state = ObsTerm(
            func=mdp.aigp_race_observation,
            params={
                "command_name": "target",
                "with_context": True,
                # Commanded thrust+attitude velocity (AI_GP BodyVelocityIntegrator).
                # Never IMU accelerometer or privileged sim twist.
                "with_velocity": True,
                # Vision stream is 30 Hz (VADR-TS-001 4.6) while control runs at
                # 60 Hz, so keypoints are held between frames. The rest of the
                # vector (attitude, body rates, commanded velocity) is IMU and
                # controller data and keeps updating every control step.
                "camera_rate_hz": 30.0,
                # How long after the shutter the keypoints actually exist:
                # sensor readout, the copy to the Orin, YOLO-pose inference.
                # In 60 Hz control steps, so 1-4 is 17-67 ms. One camera frame
                # period alone is 33 ms before inference has run at all.
                #
                # Set OBS_VERSION=v1 to switch this and the action channels off
                # together and get the exact 1632-d vector back.
                "vision_delay_steps_range": (
                    None if _OBS_VERSION == "v1" else (1, 4)
                ),
                # Observation v2. Without this the policy has no record of what
                # it commanded, so it cannot account for a command that has been
                # issued but not yet landed -- and every delay above becomes an
                # unexplainable disturbance rather than something to compensate
                # for. Eschmann's ablation: 10/10 to 0/10 with delay simulated
                # and action history removed.
                "with_actions": _OBS_VERSION != "v1",
            },
            # Stacked observations, as in AI_GP race_obs (DEFAULT_HISTORY). A
            # single frame cannot tell a gate drifting out of view from one
            # already gone, and detection falls off hard past 40 deg of pitch,
            # so the policy needs recent context to fly through a dropout.
            # At the 60 Hz control rate this spans 32 / 60 = 0.53 s.
            history_length=32,
            flatten_history_dim=True,
        )

        # Corner lookahead. Without this the keypoint channel shows only the
        # gate being flown at, so a turn is invisible until the drone is already
        # through the gate -- and measured crashes cluster on precisely the
        # sharp corners (gate 8 at 17% of all collisions, following two
        # consecutive 42 deg turns). Static course geometry, not sim state, so
        # it survives deployment. No history: it is piecewise-constant per leg.
        #
        # Set GATE_LOOKAHEAD=0 to drop it and get the exact 1632-d AI_GP vector
        # back. This is the rollback switch: it changes the observation width,
        # so checkpoints are only loadable under the setting they trained with.
        next_gate = ObsTerm(
            func=mdp.next_gate_pose_g,
            params={"command_name": "target", "scale": 30.0},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            if not int(os.environ.get("GATE_LOOKAHEAD", 0)):
                self.next_gate = None

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        image = ObsTerm(func=mdp.image)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc)
        imu_att = ObsTerm(func=mdp.imu_orientation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class PrivilegedObservationsCfg:
    """Original privileged full-state observations (position, quat, twists).

    Kept for ablation / comparison. Swap onto ``DroneRacerEnvCfg.observations``
    if you need the pre-AI_GP observation layout.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        position = ObsTerm(func=mdp.root_pos_w)
        attitude = ObsTerm(func=mdp.root_quat_w)
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic = None


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    #
    # Mass first, then inertia. The asset spawns with no mass override at all,
    # because the one MassPropertiesCfg carries would land on all five rigid
    # bodies and make the aircraft weigh five times what it does. These three
    # terms are what make the simulated airframe the documented one; without
    # them it is a 0.5 kg upstream drone wearing our thrust curve.
    body_mass = EventTerm(
        func=mdp.set_body_mass,
        mode="startup",
        params={
            "total_mass_kg": mdp.AIRFRAME_MASS_KG,
            "prop_mass_kg": mdp.PROP_MASS_KG,
        },
    )
    # The USD's authored inertia is inherited from a 0.5 kg upstream aircraft,
    # and carries products of inertia this one should not have.
    # See ``mdp.events.BODY_INERTIA_DIAG`` for how the estimate was built.
    body_inertia = EventTerm(
        func=mdp.set_body_inertia,
        mode="startup",
        params={"inertia_diag": mdp.BODY_INERTIA_DIAG},
    )
    # The props are 5 g each now rather than 1.745 kg, and PhysX does not
    # rescale inertia when mass changes -- left alone they would keep a
    # 1.745 kg link's resistance to spinning.
    prop_inertia = EventTerm(
        func=mdp.set_body_inertia,
        mode="startup",
        params={"inertia_diag": mdp.PROP_INERTIA_DIAG, "body_name": "prop.*"},
    )

    # reset
    # TODO: Resetting base happens in the command reset also for the moment
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-3.5, -1.5),
                "y": (-0.5, 0.5),
                "z": (1.5, 0.5),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # intervals
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 0.2),
        params={
            "force_range": (-0.1, 0.1),
            "torque_range": (-0.05, 0.05),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    target = mdp.GateTargetingCommandCfg(
        asset_name="robot",
        track_name="track",
        loop=True,
        randomise_start=None,
        record_fpv=False,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Weights are set for a policy that cannot yet reach a gate. The sparse
    # gate_passed bonus is the real objective, but it is unreachable early, so
    # the dense terms have to carry the first stage of learning and the crash
    # penalty must not drown them out.
    #
    # -500 on termination was ~100x every dense term combined, so the only
    # gradient the agent could find was "avoid dying", and with no signal worth
    # following the entropy bonus inflated the policy std to its ceiling.
    terminating = RewTerm(func=mdp.is_terminated, weight=-100.0)
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-0.0001)
    # 60 Hz × 40 s = 2400 steps → −24 over a full episode, ~6% of one pass.
    # A 1 s crash pays −0.6, so this cannot beat the −100 terminate (no
    # suicide-to-stop-the-clock).
    time_penalty = RewTerm(func=mdp.time_penalty, weight=-0.01)
    progress = RewTerm(func=mdp.progress, weight=20.0, params={"command_name": "target"})
    # 11→1 wrap: extra pull to the 3 m run-in. ``progress`` / proximity /
    # vel_align now aim there too when already in front of the opening.
    run_in_progress = RewTerm(
        func=mdp.run_in_progress, weight=25.0, params={"command_name": "target"}
    )
    # Pass-only. A miss no longer shares this term (see ``mdp.gate_passed``).
    # One clean hole has to dominate a full episode of look / proximity.
    gate_passed = RewTerm(func=mdp.gate_passed, weight=600.0, params={"command_name": "target"})
    # Kaufmann / Nature 2023 velocity-to-gate. Dense is direction only.
    # 0.12: a 40 s aligned episode is +288, still under one pass (600).
    vel_align_gate = RewTerm(func=mdp.vel_align_gate, weight=0.12, params={"command_name": "target"})
    # Zeroed for the completion-scored objective. This term pays directly for
    # speed through the opening, which is the behaviour that was producing gate
    # strikes at 14 m/s. ``center_gate_passage`` still pays for going through
    # cleanly, so the shaping that matters survives.
    vel_gate_passage = RewTerm(func=mdp.vel_gate_passage, weight=0.0, params={"command_name": "target"})
    # Speed above what two laps actually need, squared. At the 8 m/s cap two
    # laps of the 106 m course take 27 s of the 40 s episode -- comfortable
    # margin -- while the policy that trained without this flew at 14 m/s.
    over_speed = RewTerm(
        func=mdp.over_speed, weight=-0.5,
        params={"cap_mps": mdp.DEFAULT_SPEED_CAP_MPS},
    )
    # Smaller sibling at the crossing: FPV yaw within ψ of the gate normal.
    # 1 on-axis → +4; 0 at 40° (and beyond). Does not beat a 5 m/s vel bonus.
    cam_gate_passage = RewTerm(
        func=mdp.cam_gate_passage, weight=4.0, params={"command_name": "target", "psi_deg": 40.0}
    )
    # How centered the hole was, linear in plane offset (not a yes/no).
    # weight * (1 - r_⊥ / 0.75): +4 on the nail, +2 at 0.375 m, 0 on a clip.
    center_gate_passage = RewTerm(
        func=mdp.center_gate_passage, weight=4.0, params={"command_name": "target"}
    )
    # Standing gradient toward the centre of the opening, so an agent that has
    # never passed a gate still has something to climb. std=8 m spans about one
    # 22 m leg, giving a usable slope from the moment the previous gate is behind.
    #
    # Cut from 2.0 once the policy could reliably reach gates. This term peaks
    # when *sitting* on the gate centre, and at 2.0 that paid nearly as well as
    # closing at 10 m/s (~3.3/step from `progress`), i.e. it was a standing
    # invitation to loiter. It stays on, weakly, to keep guiding the later gates
    # the policy has not learned yet.
    gate_proximity = RewTerm(
        func=mdp.gate_proximity, weight=0.5, params={"command_name": "target", "std": 8.0}
    )
    # Look terms are nudges only. A 40 s stare at 60 Hz must not beat one
    # +400 pass — that is what the last two rates runs farmed.
    lookat_next = RewTerm(func=mdp.lookat_next_gate, weight=0.15, params={"command_name": "target", "std": 0.5})
    heading_to_gate = RewTerm(
        func=mdp.heading_to_gate, weight=0.15, params={"command_name": "target", "std": 0.35}
    )
    gate_visible = RewTerm(
        func=mdp.gate_visible, weight=0.05, params={"command_name": "target", "std": 0.55}
    )
    # High-then-low (gate 6 at 4.2 m → gate 7 at 2.0 m, and the same pattern
    # later): reward the dive — tilt forward, cut collective, actually sink —
    # instead of carrying a level high-speed line through the upper opening.
    low_gate_dive = RewTerm(
        func=mdp.low_gate_dive, weight=1.5, params={"command_name": "target"}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Reaching the last gate ends the run. Flagged as a time-out so it lands in
    # the truncation buffer: the `terminating` reward penalises only genuine
    # terminations, and finishing the course should not be punished.
    course_finished = DoneTerm(func=mdp.course_finished, params={"command_name": "target"}, time_out=True)
    # Measured against the *target* gate, so this has to clear the longest gate
    # spacing on the track (22.9 m) with enough room for a wide racing line.
    flyaway = DoneTerm(func=mdp.flyaway, params={"command_name": "target", "distance": 20.0})
    # Crossing the gate plane outside the opening used to only pay -400 and
    # keep the same target. Reset here so a fast miss is a failed episode.
    gate_missed = DoneTerm(func=mdp.gate_missed, params={"command_name": "target"})
    collision = DoneTerm(
        func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 0.01}
    )
    # PhysX / distortion can emit a NaN pose. Reset that env before PPO sees it.
    nonfinite_state = DoneTerm(func=mdp.nonfinite_state)


@configclass
class DroneRacerEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=4096, env_spacing=0.0)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        # Disable IMU and Tiled Camera
        self.scene.imu = None
        self.scene.tiled_camera = None

        # MDP settings
        self.observations.critic = None
        self.events.reset_base = None
        self.commands.target.randomise_start = True
        # A quarter of episodes begin on the floor at the run-in, the rest
        # arrive already moving.
        #
        # The race starts at G1 from a standstill. Training used the standing
        # run-in only when the target was index 0, so the policy saw a standing
        # start at one gate out of eleven -- and the best racing policy we have,
        # averaging 9.5 gates, fails 100% of episodes at G1 from the real start
        # position. It had never been there.
        #
        # A quarter rather than all of them: the other three quarters are what
        # teach carrying speed between gates, which is the rest of the race.
        self.commands.target.floor_start_fraction = 0.25
        # 0.12 m is wheels-down without touching: the collision termination
        # fires at 0.01 N, so an aircraft resting on the floor would end its
        # episode before the policy acted. Up to 0.5 m covers a hand launch.
        self.commands.target.spawn_z_range = (0.12, 0.50)

        # general settings
        # VADR-TS-001 3.2/4.4: physics runs at 120 Hz and the command rate must
        # be *below* 100 Hz. Decimating by 2 gives a 60 Hz policy, which clears
        # that bound outright (AI_GP picked 99 Hz for the same reason) while
        # matching the simulator's physics rate exactly.
        self.decimation = 2
        # The 379 m course is ~25 s at racing speed; a developing policy is much
        # slower, so leave headroom for it to reach the finish gate.
        self.episode_length_s = 40
        # viewer settings: framed over the 21 × 50 m PQ hall
        self.viewer.eye = (10.5, -35.0, 32.0)
        self.viewer.lookat = (10.5, 25.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@configclass
class DroneRacerEnvCfg_PLAY(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=4096, env_spacing=0.0)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        self.scene.imu = None
        # The FPV camera exists only for looking at the run; the policy flies on
        # projected keypoints and never reads a pixel, so rendering it is pure
        # cost. Spawning it when the app was launched without cameras makes Isaac
        # Lab abort, so keep it tied to the same ENABLE_CAMERAS switch the
        # AppLauncher reads. Training never sets it.
        if not int(os.environ.get("ENABLE_CAMERAS", 0)):
            self.scene.tiled_camera = None

        # MDP settings
        self.observations.critic = None

        # Watching a run is only meaningful start-to-finish. Training randomises
        # the entry gate so every leg gets visited, but here that just teleports
        # the drone to an arbitrary point on the course each reset.
        self.commands.target.randomise_start = False
        # Spawn by the organizer PDF number, not Isaac collection index.
        # Index 0 is official G6; official G1 is later in the list.
        self.commands.target.official_start_gate = int(os.environ.get("PLAY_START_OFFICIAL", "1"))
        self.commands.target.fixed_start_idx = int(os.environ.get("PLAY_START_IDX", "6"))

        # Disable push robot events. Spawn is the command run-in, same as train.
        self.events.push_robot = None
        self.events.reset_base = None

        # Enable recording fpv footage
        # self.commands.target.record_fpv = True

        # PLAY-only general settings
        # VADR-TS-001 3.2/4.4: physics runs at 120 Hz and the command rate must
        # be *below* 100 Hz. Decimating by 2 gives a 60 Hz policy, which clears
        # that bound outright (AI_GP picked 99 Hz for the same reason) while
        # matching the simulator's physics rate exactly.
        self.decimation = 2
        # The 379 m course is ~25 s at racing speed; a developing policy is much
        # slower, so leave headroom for it to reach the finish gate.
        self.episode_length_s = 40
        # viewer settings: chase the drone rather than framing the whole course.
        # A static view wide enough to hold 269 m of track renders the 280 mm
        # airframe at roughly one pixel, so the run looks empty. Anchoring to the
        # asset root makes eye/lookat offsets relative to the drone.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-4.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
