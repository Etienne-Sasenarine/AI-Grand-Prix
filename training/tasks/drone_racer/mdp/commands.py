# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import cv2
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass

from utils.vel_align import cam_horizontal_align, center_passage_score

from .events import reset_after_prev_gate

# Organizer PDF: 85x165 ft, origin top-left, Y down the sheet.
_PDF_FT = {
    1: (12.0, 71.0),
    2: (16.0, 39.0),
    3: (40.0, 17.0),
    4: (72.0, 40.0),
    5: (66.0, 70.0),
    6: (41.0, 86.0),
    7: (70.5, 106.6),
    8: (60.0, 129.0),
    9: (39.7, 147.7),
    10: (13.0, 110.0),
}


def _official_xy_m(official_n: int) -> tuple[float, float]:
    x_ft, y_ft = _PDF_FT[int(official_n)]
    return x_ft * 0.3048, (165.0 - y_ft) * 0.3048


def official_number_from_xy(x: float, y: float) -> int:
    best_n, best_d = 1, float("inf")
    for n, (x_ft, y_ft) in _PDF_FT.items():
        ox, oy = x_ft * 0.3048, (165.0 - y_ft) * 0.3048
        d = (x - ox) ** 2 + (y - oy) ** 2
        if d < best_d:
            best_d, best_n = d, n
    return best_n


def isaac_index_for_official(gate_xy, official_n: int) -> int:
    tx, ty = _official_xy_m(official_n)
    best_i, best_d = 0, float("inf")
    for i in range(len(gate_xy)):
        dx = float(gate_xy[i][0]) - tx
        dy = float(gate_xy[i][1]) - ty
        d = dx * dx + dy * dy
        if d < best_d:
            best_d, best_i = d, i
    return best_i

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GateTargetingCommand(CommandTerm):
    """Command generator that generates a pose command from a uniform distribution."""

    cfg: GateTargetingCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: GateTargetingCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self.cfg = cfg

        # FPV video recording
        if self.cfg.record_fpv:
            self.video_id = 0
            self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera")
            self.sensor: TiledCamera = self._env.scene.sensors[self.sensor_cfg.name]

        # extract the robot and track for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.track: RigidObjectCollection = env.scene[cfg.track_name]
        self.gate_size = cfg.gate_size
        self.num_gates = self.track.num_objects

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in simulation world frame
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.prev_robot_pos_w = self.robot.data.root_pos_w.clone()
        self._gate_missed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._course_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_pass_speed = torch.zeros(self.num_envs, device=self.device)
        self._gate_pass_cam_align = torch.zeros(self.num_envs, device=self.device)
        self._gate_pass_center = torch.zeros(self.num_envs, device=self.device)
        self.next_gate_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.next_gate_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._lookahead_gate_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._resolved_play_start: int | None = None

    def __str__(self) -> str:
        msg = "GateTargetingCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.next_gate_w

    @property
    def gate_missed(self) -> torch.Tensor:
        return self._gate_missed

    @property
    def gate_passed(self) -> torch.Tensor:
        return self._gate_passed

    @property
    def course_completed(self) -> torch.Tensor:
        """True once the final gate of a non-looping course has been passed."""
        return self._course_completed

    @property
    def lookahead_gate(self) -> torch.Tensor:
        """Pose of the gate after the current target. Shape is (num_envs, 7)."""
        return self._lookahead_gate_w

    @property
    def gate_pass_speed(self) -> torch.Tensor:
        """Speed along the gate normal on the step a gate was passed, else zero."""
        return self._gate_pass_speed

    @property
    def gate_pass_cam_align(self) -> torch.Tensor:
        """Horizontal camera-to-normal ``max(0, ĉam_xy · n̂)`` on a pass, else 0."""
        return self._gate_pass_cam_align

    @property
    def gate_pass_center(self) -> torch.Tensor:
        """1 at the opening centre, 0 at the rim, on a pass; else 0."""
        return self._gate_pass_center

    @property
    def previous_pos(self) -> torch.Tensor:
        return self.prev_robot_pos_w

    def official_number(self, isaac_idx: int) -> int:
        pos = self.track.data.object_com_pos_w[0, int(isaac_idx)]
        return official_number_from_xy(float(pos[0]), float(pos[1]))

    def _play_start_idx(self) -> int:
        official = getattr(self.cfg, "official_start_gate", None)
        if official is None:
            raw = os.environ.get("PLAY_START_OFFICIAL", "").strip()
            official = int(raw) if raw else None
        if official is not None:
            # Isaac collection order matches env_cfg insertion: 0=G6 … 6=G1.
            # Do not search live tensors here — first resample can run before
            # gate poses are written, which used to snap every start to idx 0.
            static = {1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 0, 7: 1, 8: 2, 9: 3, 10: 5}
            idx = int(static.get(int(official), self.cfg.fixed_start_idx))
            self._resolved_play_start = idx % max(self.num_gates, 1)
            return int(self._resolved_play_start)
        return int(self.cfg.fixed_start_idx) % max(self.num_gates, 1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # Release and reinitialize video writer only after the first iteration
        if hasattr(self, "out") and self.cfg.record_fpv:
            self.out.release()
            print(f"FPV video saved as fpv_{self.video_id}.mp4")
            self.video_id += 1

        if self.cfg.record_fpv:
            self.out = cv2.VideoWriter(f"fpv_{self.video_id}.mp4", self.fourcc, 100, (1000, 1000))

        self._course_completed[env_ids] = False

        official = getattr(self.cfg, "official_start_gate", None)
        if official is None:
            raw = os.environ.get("PLAY_START_OFFICIAL", "").strip()
            official = int(raw) if raw else None
        place_drone = official is not None or self.cfg.randomise_start is not None

        if not place_drone:
            self.next_gate_idx[env_ids] = 0

        else:
            if official is None and self.cfg.randomise_start:
                self.next_gate_idx[env_ids] = torch.randint(
                    low=0, high=self.num_gates, size=(len(env_ids),), device=self.device, dtype=torch.int32
                )
            else:
                start_idx = self._play_start_idx()
                self.next_gate_idx[env_ids] = start_idx

            gate_indices = self.next_gate_idx - 1
            gate_positions = self.track.data.object_com_pos_w[self.env_ids, gate_indices]
            gate_orientations = self.track.data.object_quat_w[self.env_ids, gate_indices]

            # 3 m run-in along ``-n̂`` of the target: play's chosen gate, or
            # any train reset that starts on gate 1 (do not wrap to the last
            # gate). reset_after_prev_gate then adds +1 m.
            target_pos = self.track.data.object_com_pos_w[self.env_ids, self.next_gate_idx]
            target_quat = self.track.data.object_quat_w[self.env_ids, self.next_gate_idx]
            back = math_utils.quat_apply(
                target_quat,
                torch.tensor(
                    [-(self.cfg.start_run_in_m + 1.0), 0.0, 0.0], device=self.device
                ).expand(self.num_envs, 3),
            )
            # The start the race actually begins from, for a slice of episodes.
            #
            # Marked on the course map: roughly 4 m past G10 on the G10->G1
            # leg, on the floor, 7.8 m short of G1 and very nearly on its
            # through-axis. That is a longer run-in to one specific gate than
            # the generic 3 m run-in below, and it is the only start that will
            # ever happen in competition.
            #
            # Scattered by ``race_start_scatter_m`` because nobody sets a
            # takeoff pad down to the centimetre, and because a policy trained
            # on one exact coordinate learns that coordinate rather than the
            # approach.
            race_mask = None
            if self.cfg.race_start_xy is not None and self.cfg.race_start_fraction > 0.0:
                # Draw for the *resetting* envs only, and write only their
                # rows. next_gate_idx is persistent state: assigning the whole
                # tensor re-targets every env in flight, so an aircraft two
                # gates into a lap is suddenly told it is aiming at G1. The
                # file already carries a warning about this exact mistake --
                # "Only the reset envs: overwriting every env here erased the
                # crossing check for all envs on any step with a reset (Sep 6
                # run collapse)" -- and it was made again anyway.
                race_mask = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
                sel = (
                    torch.rand(len(env_ids), device=self.device)
                    < self.cfg.race_start_fraction
                )
                race_mask[env_ids] = sel
                if bool(sel.any()):
                    start = int(self.cfg.race_start_idx)
                    picked = self.next_gate_idx[env_ids]
                    self.next_gate_idx[env_ids] = torch.where(
                        sel, torch.full_like(picked, start), picked
                    )

            if self.cfg.floor_start_fraction > 0.0:
                # A mixture, not a switch.
                #
                # Training used the stationary run-in only when the target was
                # index 0, so the policy met that geometry at exactly one gate
                # out of eleven. Play forces it at whatever gate you start
                # from, and the real race starts at G1 -- which is how a policy
                # averaging 9.5 gates came to fail 100% of episodes at the
                # first gate, having never once seen a standing start there.
                #
                # Making every episode a standing start would fix that and
                # break something else: the policy would stop practising the
                # continuous, already-moving course flying that the other ten
                # gates need. So a fraction start from rest and the rest carry
                # speed through, and both stay in distribution.
                at_run_in = (
                    torch.rand(self.num_envs, 1, device=self.device)
                    < self.cfg.floor_start_fraction
                )
            elif self.cfg.start_at_run_in or not self.cfg.randomise_start:
                # Every episode begins in front of its target gate.
                at_run_in = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
            else:
                at_run_in = (self.next_gate_idx == 0).unsqueeze(-1)
            gate_positions = torch.where(at_run_in, target_pos + back, gate_positions)
            gate_orientations = torch.where(at_run_in, target_quat, gate_orientations)

            # Face the drone at the gate it has to fly through, not down the
            # previous gate's normal. The two differ by the whole turn angle at a
            # corner, and with a 90 deg horizontal FoV that difference is enough
            # to start the episode with the target off-frame -- no keypoints, so
            # nothing for the policy to servo on. Aim from where the drone will
            # actually appear, which is 1 m past the previous gate (the offset
            # reset_after_prev_gate applies).
            # Scatter the spawn *before* the heading is computed, not after.
            #
            # reset_after_prev_gate applies pose_range jitter on top of whatever
            # pose it is handed, and the heading below is derived from the
            # un-jittered position -- so widening that jitter aims the camera
            # away from the gate by up to the scatter angle. At 2 m out a 3 m
            # sideways offset is 56 degrees, well outside the 72.8 degree
            # horizontal frame: the gate leaves view and the observation goes
            # blind, which is the one thing a perception-driven spawn must not
            # do.
            #
            # Applying it here instead means the heading is aimed from where
            # the aircraft will actually appear, so the gate is in frame by
            # construction however wide the scatter gets.
            if self.cfg.spawn_scatter_m is not None:
                sx, sy, sz = self.cfg.spawn_scatter_m
                spread = torch.tensor([sx, sy, sz], device=self.device)
                offset = (torch.rand(self.num_envs, 3, device=self.device) * 2.0 - 1.0) * spread
                gate_positions = gate_positions + offset

            # Sample the start height, after any scatter and before the
            # heading is aimed.
            #
            # The offset reset_after_prev_gate applies is along the gate
            # normal, horizontal for an upright gate, so setting z here
            # survives it: the aircraft appears below where it is meant to end
            # up and has to climb.
            #
            # A range rather than the floor, for two reasons. It keeps the
            # bottom of the range clear of the ground -- the collision
            # termination fires at 0.01 N, so an aircraft resting on the floor
            # ends its episode before the policy has acted. And it varies how
            # far there is to climb, so the policy learns to arrive at a height
            # rather than to perform one fixed ascent.
            if race_mask is not None and bool(race_mask.any()):
                sx, sy = self.cfg.race_start_xy
                sc = float(self.cfg.race_start_scatter_m)
                jitter = (torch.rand(self.num_envs, 2, device=self.device) * 2.0 - 1.0) * sc
                gate_positions = gate_positions.clone()
                gate_positions[:, 0] = torch.where(
                    race_mask, sx + jitter[:, 0], gate_positions[:, 0])
                gate_positions[:, 1] = torch.where(
                    race_mask, sy + jitter[:, 1], gate_positions[:, 1])
                # These start on the floor whatever the run-in draw said, and
                # the +1 m normal offset reset_after_prev_gate adds would push
                # them off the pad, so cancel it for these envs only.
                back_off = math_utils.quat_apply(
                    gate_orientations,
                    torch.tensor([-1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3),
                )
                gate_positions = torch.where(
                    race_mask.unsqueeze(-1), gate_positions + back_off, gate_positions)
                at_run_in = at_run_in | race_mask.unsqueeze(-1)

            if self.cfg.spawn_z_range is not None:
                lo, hi = self.cfg.spawn_z_range
                gate_positions = gate_positions.clone()
                sampled = torch.rand(self.num_envs, device=self.device) * (hi - lo) + lo
                # Only the standing starts go to the floor. An env that is
                # meant to arrive already moving keeps the height its previous
                # gate implies -- dropping it to the floor as well would erase
                # the in-motion start this mixture exists to preserve.
                on_floor = at_run_in.squeeze(-1)
                gate_positions[:, 2] = torch.where(
                    on_floor, sampled, gate_positions[:, 2]
                )

            # Freeze the spawn pose *here*, after every block that moves the
            # aircraft, not before them.
            #
            # torch.cat copies. Building gate_w above the scatter, the
            # competition pad and the height range left reset_after_prev_gate
            # holding a snapshot none of those three had touched, so all three
            # reached only spawn_pos -- which feeds the heading and nothing
            # else. The aircraft was aimed from the pad and placed at the gate
            # run-in: measured, 0 of 1024 spawns within 1.5 m of the pad and
            # none below 0.67 m, against a configured 20% and 32%.
            #
            # Nothing warns when a tensor stops aliasing. The check that does
            # is reading the position back out of the simulation, which is what
            # scripts/diag_spawn.py exists to do.
            gate_w = torch.cat([gate_positions, gate_orientations], dim=1)

            spawn_pos = gate_positions + math_utils.quat_apply(
                gate_orientations,
                torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3),
            )
            target_pos = self.track.data.object_com_pos_w[self.env_ids, self.next_gate_idx]
            aim = target_pos - spawn_pos
            aim_yaw = torch.atan2(aim[:, 1], aim[:, 0])
            zeros = torch.zeros_like(aim_yaw)
            heading_quat = math_utils.quat_from_euler_xyz(zeros, zeros, aim_yaw)

            play_start = not self.cfg.randomise_start
            # Position jitter is already applied above when spawn_scatter_m is
            # set; applying it twice would reintroduce the heading error the
            # scatter block exists to avoid. Attitude jitter is unaffected.
            scattered = self.cfg.spawn_scatter_m is not None
            xy = 0.0 if (play_start or scattered) else float(self.cfg.reset_pos_xy_m)
            z = 0.0 if (play_start or scattered) else float(self.cfg.reset_pos_z_m)
            rp = 0.0 if play_start else float(self.cfg.reset_roll_pitch_rad)
            yw = 0.0 if play_start else float(self.cfg.reset_yaw_rad)
            reset_after_prev_gate(
                env=self._env,
                env_ids=env_ids,
                gate_pose=gate_w,
                heading_quat=heading_quat,
                # Play: exact 3 m run-in, no scatter. Train: start-gate
                # curriculum plus location noise and a small upright
                # attitude jitter (a few degrees, not on its side).
                # Interval pushes already supply acceleration; spawn
                # velocity stays zero so those hits stay the accel source.
                pose_range={
                    "x": (-xy, xy),
                    "y": (-xy, xy),
                    "z": (-z, z),
                    "roll": (-rp, rp),
                    "pitch": (-rp, rp),
                    "yaw": (-yw, yw),
                },
                velocity_range={
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                asset_cfg_name=self.cfg.asset_name,
            )

        # The reset teleport is not a plane crossing. Keep prev on this side
        # so the first physics step cannot count the start gate as already passed.
        # Only the reset envs: overwriting every env here erased the crossing
        # check for all envs on any step with a reset (Sep 6 run collapse).
        # Clone first: _update_command stores root_pos_w without copying.
        prev = self.prev_robot_pos_w.clone()
        prev[env_ids] = self.robot.data.root_pos_w[env_ids]
        self.prev_robot_pos_w = prev
        self._gate_passed[env_ids] = False
        self._gate_missed[env_ids] = False

        # The index just changed and the drone has just been teleported. Publish
        # the matching pose now rather than leaving the previous episode's gate
        # standing until the next ``_update_command``, which the step loop does
        # not reach until after terminations have already been judged against it.
        self._refresh_gate_poses()

    def _refresh_gate_poses(self):
        """Republish the target and lookahead poses from ``next_gate_idx``.

        Called from both ``_resample_command`` and ``_update_command``, because
        the index and the pose must never disagree.

        They used to. Only ``_update_command`` wrote these, and Isaac Lab runs
        ``command_manager.compute()`` *after* the termination and reward
        managers. So for exactly one step following an explicit ``env.reset()``
        the pose described the previous episode's gate while the drone had
        already been teleported somewhere else on the course -- and on a
        freshly built env it was still the (0, 0, 0) these buffers are
        initialised to. ``flyaway`` measures distance to this pose and fired
        for 48 of 64 envs on step one, at zero speed, before the policy had
        done anything at all.

        Idempotent and cheap: two gathers, no state beyond the two buffers.
        """
        next_gate_positions = self.track.data.object_com_pos_w[self.env_ids, self.next_gate_idx]
        next_gate_orientations = self.track.data.object_quat_w[self.env_ids, self.next_gate_idx]
        self.next_gate_w = torch.cat([next_gate_positions, next_gate_orientations], dim=1)

        # Pose of the gate *after* the current target, so the policy can see a
        # corner before it is committed to one. Computed from the same index as
        # ``next_gate_w`` above, before the pass check advances it, so the two
        # always describe consecutive gates.
        if self.cfg.loop:
            lookahead_idx = (self.next_gate_idx + 1) % self.num_gates
        else:
            lookahead_idx = (self.next_gate_idx + 1).clamp(max=self.num_gates - 1)
        self._lookahead_gate_w = torch.cat(
            [
                self.track.data.object_com_pos_w[self.env_ids, lookahead_idx],
                self.track.data.object_quat_w[self.env_ids, lookahead_idx],
            ],
            dim=1,
        )

    def _update_command(self):
        if self.cfg.record_fpv:
            image = self.sensor.data.output["rgb"][0].cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.out.write(image)

        self._refresh_gate_poses()

        # Gate passing logic
        (roll, pitch, yaw) = math_utils.euler_xyz_from_quat(self.next_gate_w[:, 3:7])
        normal = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        pos_old_projected = (self.prev_robot_pos_w[:, 0] - self.next_gate_w[:, 0]) * normal[:, 0] + (
            self.prev_robot_pos_w[:, 1] - self.next_gate_w[:, 1]
        ) * normal[:, 1]
        pos_new_projected = (self.robot.data.root_pos_w[:, 0] - self.next_gate_w[:, 0]) * normal[:, 0] + (
            self.robot.data.root_pos_w[:, 1] - self.next_gate_w[:, 1]
        ) * normal[:, 1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)

        self._gate_passed = passed_gate_plane & (
            torch.all(torch.abs(self.robot.data.root_pos_w - self.next_gate_w[:, :3]) < (self.gate_size / 2), dim=1)
        )

        self._gate_missed = passed_gate_plane & (
            torch.any(torch.abs(self.robot.data.root_pos_w - self.next_gate_w[:, :3]) > (self.gate_size / 2), dim=1)
        )

        # Speed through the opening, along the gate normal, latched on the step
        # of the crossing. Rewards run after this update, by which point
        # ``next_gate_idx`` has already advanced and the normal of the gate that
        # was actually passed is no longer recoverable. Projecting onto the
        # normal rather than using |v| means a fast sideways drift through the
        # opening does not read as a fast gate.
        vel_w = self.robot.data.root_lin_vel_w
        self._gate_pass_speed = torch.where(
            self._gate_passed,
            vel_w[:, 0] * normal[:, 0] + vel_w[:, 1] * normal[:, 1],
            torch.zeros_like(vel_w[:, 0]),
        )

        # Camera look-through, latched on the same step for the same reason:
        # the target gate advances below and its yaw would be lost.
        n_hat = torch.stack(
            [normal[:, 0], normal[:, 1], torch.zeros_like(normal[:, 0])], dim=-1
        )
        self._gate_pass_cam_align = torch.where(
            self._gate_passed,
            cam_horizontal_align(self.robot.data.root_quat_w, n_hat),
            torch.zeros_like(vel_w[:, 0]),
        )
        self._gate_pass_center = torch.where(
            self._gate_passed,
            center_passage_score(
                self.robot.data.root_pos_w,
                self.next_gate_w[:, :3],
                n_hat,
                half_size=self.gate_size / 2,
            ),
            torch.zeros_like(vel_w[:, 0]),
        )

        # Update next gate target for the envs that passed the gate
        self.next_gate_idx[self._gate_passed] += 1
        if self.cfg.loop:
            self.next_gate_idx = self.next_gate_idx % self.num_gates
        else:
            # Running off the end means the finish gate was passed. Latch that and
            # clamp, so the command stays in range for the step that reports it.
            self._course_completed |= self.next_gate_idx >= self.num_gates
            self.next_gate_idx = self.next_gate_idx.clamp(max=self.num_gates - 1)

        # Snapshot: root_pos_w is a view into a TimestampedBuffer. A live
        # alias would make prev==curr on the next step and kill plane-cross.
        self.prev_robot_pos_w = self.robot.data.root_pos_w.clone()

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                # -- goal pose
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                # -- current body pose
                self.drone_visualizer = VisualizationMarkers(self.cfg.drone_visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
            self.drone_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
                self.drone_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        self.target_visualizer.visualize(self.next_gate_w[:, :3], self.next_gate_w[:, 3:])
        self.drone_visualizer.visualize(self.robot.data.root_pos_w, self.robot.data.root_quat_w)


@configclass
class GateTargetingCommandCfg(CommandTermCfg):
    """Configuration for gate targeting command generator."""

    class_type: type = GateTargetingCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    track_name: str = MISSING
    """Name of the track in the environment for which the commands are generated."""

    randomise_start: bool | None = None
    """If True, the starting gate is randomised at every reset."""

    fixed_start_idx: int = 0
    """When ``randomise_start`` is False, the 0-based gate the drone starts on."""

    official_start_gate: int | None = None
    """PDF gate number (1–10). Play looks up that XY on the track instead of
    trusting Isaac's collection index. ``PLAY_START_OFFICIAL`` overrides."""

    loop: bool = True
    """If True the course is a circuit and the target wraps from the last gate back to the first.

    If False the course is point-to-point: passing the final gate sets
    :attr:`GateTargetingCommand.course_completed` instead of wrapping, which the
    ``course_finished`` termination uses to end the episode.
    """

    start_run_in_m: float = 3.0
    """Metres behind the first gate (along ``-n̂``) when an episode starts on gate 1."""

    race_start_xy: tuple[float, float] | None = None
    """Hall (x, y) of the takeoff pad the competition run actually starts from.

    The organizer spec times a run "from the first timing line cross / gate
    trigger" and says gates are flown in numerical order from Gate 1, so where
    the aircraft leaves the ground is ours to choose -- but there is exactly
    one such place per run, and the policy has to fly it.

    Set with ``race_start_fraction`` and ``race_start_idx``. Episodes drawn for
    it spawn here on the floor, targeting the start gate, whatever the rest of
    the start distribution is doing.
    """

    race_start_fraction: float = 0.0
    """Fraction of episodes beginning at ``race_start_xy``.

    Deliberately a slice, not the whole thing. The start happens once per run
    and the other twenty-one gates happen after it, so a policy that trains
    mostly on the start would be optimising the rarest part of the race.
    """

    race_start_scatter_m: float = 0.8
    """Half-width of the jitter on the pad position, metres.

    A takeoff pad is not placed to the centimetre, and a policy trained on one
    exact coordinate learns the coordinate rather than the approach.
    """

    race_start_idx: int = 6
    """Isaac collection index of the start gate. 6 is official G1."""

    floor_start_fraction: float = 0.0
    """Fraction of episodes that begin stationary at the run-in, not in motion.

    Zero keeps the original behaviour: the standing run-in happens only when
    the target is gate index 0, and every other episode spawns a metre past the
    previous gate already carrying speed.

    That is how a policy averaging 9.5 gates came to fail *every* episode at
    the first gate. The race starts at G1 from a standstill, and training had
    shown it a standstill at exactly one gate out of eleven.

    Set with ``spawn_z_range`` to make those episodes start on the floor, so
    the aircraft has to take off, climb and fly the gate -- which is what a
    real run is. Keep it below 1.0: the episodes that are *not* standing starts
    are what teach continuous course flying, and a policy trained only on
    standing starts forgets how to carry speed between gates.
    """

    spawn_z_range: tuple[float, float] | None = None
    """Uniform range to sample the spawn height from, overriding the nominal one.

    ``None`` spawns the aircraft in the air at the point it is meant to hold,
    which trains recovery-to-hover but never a climb. Set a range and the
    episode starts low: the aircraft must rise to the hold point and then keep
    it, which is what a cage test looks like when someone sets the drone down
    and arms it.

    Keep the floor of the range clear of zero. The collision termination fires
    at 0.01 N, so an aircraft resting on the ground ends its episode before the
    policy has acted at all.
    """

    spawn_scatter_m: tuple[float, float, float] | None = None
    """Half-width of a uniform box scattering the spawn, applied before heading.

    ``None`` keeps the original behaviour: spawn on the nominal point, with
    ``reset_pos_xy_m`` / ``reset_pos_z_m`` jitter added afterwards.

    Set it and the scatter happens *before* the aim heading is computed, so the
    aircraft is pointed at the gate from wherever it actually lands and the
    gate stays in frame however wide the box. That is what makes a wide spawn
    usable for a perception-driven task: the policy sees the gate from a real
    spread of positions rather than from one specific spot, which is how it
    will be handed the aircraft in a cage.

    Needs a reward with reach to be useful -- a Gaussian 0.75 m wide has none
    past a couple of metres. ``drone_hover.mdp.approach`` is the companion.
    """

    start_at_run_in: bool = False
    """Start every episode in front of the target gate, not after the previous one.

    Racing wants the default: the drone appears one metre past the gate it has
    just passed, which is where it would really be mid-course, and only an
    episode beginning on gate 1 gets the run-in.

    Station keeping wants the opposite, and needs it decoupled from
    ``randomise_start``. Its reward is a Gaussian on distance to a hold point
    two metres in front of the gate; spawning after the *previous* gate puts
    the aircraft a full gate-spacing away, where that Gaussian returns zero and
    there is no gradient to follow. Measured on the 11-gate course: 40 of 64
    envs started beyond the task's own 8 m ``flyaway`` radius and terminated on
    step one, at zero speed.

    Set alongside ``start_run_in_m`` to place the spawn exactly on the hold
    point. Scatter still applies -- this flag chooses *where* the episode
    begins, not whether the start is jittered.
    """

    reset_pos_xy_m: float = 1.0
    """Train spawn scatter in world x/y (m). Play forces 0."""

    reset_pos_z_m: float = 0.7
    """Train spawn scatter in world z (m). Play forces 0."""

    reset_roll_pitch_rad: float = 0.12
    """Train spawn roll/pitch half-range (rad, ~7 deg). Stay upright. Play forces 0."""

    reset_yaw_rad: float = 0.15
    """Train spawn yaw half-range (rad, ~9 deg). Slight heading noise. Play forces 0."""

    record_fpv: bool = False
    """If True, the first-person view (FPV) camera is recorded during the simulation."""

    gate_size: float = 1.5
    """Size of the gate in meters. This is used to determine if the drone has passed through the gate."""

    target_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    drone_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    target_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
    drone_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
