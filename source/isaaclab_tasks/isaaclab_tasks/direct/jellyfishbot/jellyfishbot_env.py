# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go-to-position task for the IADYS Jellyfishbot.

The task follows the Kingfisher one (``direct/kingfisher``): same observations, rewards, terminations and resets. The
robot has three Blue Robotics T200 thrusters (left, right and a lateral one) and the hydrodynamics of its gz-sim model
(``jellyfishbot_simulation/jfb_gazebo/urdf/jfb_gazebo_plugins.xacro`` and ``jfb_sim/cmd_drive_translate.py``).
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuator_force.actuator_force import PropellerActuator, PropellerActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.physics.hydrodynamics import Hydrodynamics, HydrodynamicsCfg
from isaaclab.physics.hydrostatics import Hydrostatics, HydrostaticsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets.robots.jellyfishbot import JELLYFISHBOT_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class JellyfishbotEnvWindow(BaseEnvWindow):
    """Window manager for the Jellyfishbot environment."""

    def __init__(self, env: JellyfishbotEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class JellyfishbotEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 60.0
    physics_dt = 1 / 50.0  # 50 Hz
    decimation = 5
    step_dt = physics_dt * decimation  # 10 Hz
    action_space = 3  # left, right, lateral thruster commands in [-1, 1]
    observation_space = 9
    state_space = 0
    debug_vis = True

    ui_window_class_type = JellyfishbotEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=True,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = JELLYFISHBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Thruster bodies, in the order of the actions. Thrust acts along each link's +x: forward for left/right, to port
    # for lateral (its link is yawed by pi/2).
    thruster_body_names = ["left_thruster_link", "right_thruster_link", "lateral_thruster_link"]

    # Hydrostatics
    # jfb_gazebo (vrx::Surface) rests with base_link 0.20 m above the water, with the effective keel at z=-0.255 in
    # base_link. The box model gives the same equilibrium: draught 0.055 m * 0.328 m^2 * 997 kg/m^3 = 18.0 kg.
    hydrostatics_cfg: HydrostaticsCfg = HydrostaticsCfg(
        mass=18.0,  # Kg, total mass of the USD (16.98 base + 3 x 0.34 T200 + sensor frames). Not used by the forces.
        width=0.68,  # m, CAD
        length=0.69,  # m, CAD
        waterplane_area=0.328,  # m^2, effective (see above)
        draught_offset=0.255,  # m, base_link to the effective keel (jfb_surface_keel_z)
        max_draught=0.16,  # m, effective keel to the top of the floaters
        average_hydrostatics_force=176.6,  # N, m * g
        amplify_torque=2.0,  # Matches the Gazebo pitch (~0.06 rad bow-up at full forward); check_dynamics.py checks it
    )

    # Hydrodynamics: vrx::SimpleHydrodynamics of jfb_gazebo, all [est] there
    # [u (Forward), v (Lateral), w (Vertical), p (Roll), q (Pitch), r (Yaw)]
    # Full command in Gazebo: 1.02 m/s forward, 0.89 m/s reverse, 0.56 m/s sway, 1.97 rad/s yaw.
    hydrodynamics_cfg: HydrodynamicsCfg = HydrodynamicsCfg(
        linear_damping=[10.0, 20.0, 50.0, 10.0, 20.0, 1.0],  # xU, yV, zW, kP, mQ, nR
        quadratic_damping=[60.0, 80.0, 20.0, 10.0, 20.0, 4.0],  # xUU, yVV, zWW, kPP, mQQ, nRR
        use_drag_randomization=False,
        linear_damping_rand=[0.1, 0.1, 0.0, 0.0, 0.0, 0.1],
        quadratic_damping_rand=[0.1, 0.1, 0.0, 0.0, 0.0, 0.1],
    )
    # Drag acts at Gazebo's hydro_link (submerged hull centroid), in base_link coordinates. Forces from the wrench
    # composer act at the CoM (z=-0.19), so the env adds the lever torque. Without it the pitch arm of the thrust
    # (z=-0.384) against the drag would be 0.19 m instead of 0.10 m.
    hydro_point_b = (0.0, 0.0, -0.28)

    # Thruster dynamics: Blue Robotics T200 at 12 V (cmd_drive_translate.py), PWM 1500 + 400 * command
    propeller_cfg: PropellerActuatorCfg = PropellerActuatorCfg(
        cmd_lower_range=-1.0,
        cmd_upper_range=1.0,
        command_rate=10.0,  # max change of the command per second (0 -> full in 0.1 s)
        forces=[
            -28.48,  # -1.0
            -25.13,  # -0.9
            -21.0,  # -0.8
            -16.99,  # -0.7
            -13.43,  # -0.6
            -10.01,  # -0.5
            -7.07,  # -0.4
            -4.31,  # -0.3
            -2.0,  # -0.2
            -0.36,  # -0.1
            0.0,  # 0.0
            0.4,  # 0.1
            2.49,  # 0.2
            5.38,  # 0.3
            8.76,  # 0.4
            12.54,  # 0.5
            16.77,  # 0.6
            21.35,  # 0.7
            26.6,  # 0.8
            31.98,  # 0.9
            36.42,  # 1.0
        ],
        interp_resolution=1001,
        enable_randomization=True,
        randomization_range=0.1,
        enable_init_randomization=True,
    )

    max_energy = 3.0  # Max of 1.0 per thruster

    # Thresholds
    goal_reached_threshold = 0.1
    velocity_lower_bound = -0.05
    velocity_upper_bound = 0.6
    bearing_reached_threshold = 0.1

    # Reward scales
    distance_progress_reward_scale = 1.0
    bearing_progress_reward_scale = 5.0
    goal_reached_scale = 10.0
    bearing_reached_reward_scale = 0.01
    time_penalty_scale = -0.05
    energy_penalty_scale = -0.1
    # Off: no speed limit, the boat may go as fast as the thrusters allow (the Kingfisher uses 1.0, with the
    # velocity_lower_bound/velocity_upper_bound band on vx). The term is still computed and logged as 6_velocity.
    velocity_penalty_scale = 0.0

    # Reward coefficients
    velocity_penalty_coef = -10.0
    bearing_progress_coef = -1.5

    # Environment
    min_target_distance = 1.0
    max_target_distance = 10.0
    min_target_bearing = -torch.pi
    max_target_bearing = torch.pi

    # Enable randomizations
    enable_v0_randomizations = True
    enable_com_randomization = True
    enable_external_wrench_randomization = True


class JellyfishbotEnv(DirectRLEnv):
    cfg: JellyfishbotEnvCfg

    def __init__(self, cfg: JellyfishbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "1_distance_progress",
                "2_bearing_progress",
                "3_goal_reached",
                "4_bearing_reached",
                "5_energy",
                "6_velocity",
                "7_time",
            ]
        }
        # Get specific body indices
        self._base_link = self._robot.find_bodies("base_link")[0]
        self._thruster_ids = self._robot.find_bodies(self.cfg.thruster_body_names, preserve_order=True)[0]
        self._num_thrusters = len(self._thruster_ids)
        if self._num_thrusters != len(self.cfg.thruster_body_names):
            raise ValueError(f"Expected thruster bodies {self.cfg.thruster_body_names}, found {self._num_thrusters}.")

        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Centre of mass of base_link in its link frame. The defaults are kept so the randomization doesn't accumulate.
        self._default_coms = self._robot.root_physx_view.get_coms().clone()  # (num_envs, num_bodies, 7), on CPU
        self._base_com_b = self._default_coms[:, self._base_link[0], :3].to(self.device)
        self._hydro_point_b = torch.tensor(self.cfg.hydro_point_b, device=self.device).repeat(self.num_envs, 1)

        # Forces
        self._hydrodynamic_force = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self._hydrostatic_force = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self._thruster_forces = torch.zeros(self.num_envs, self._num_thrusters, 3, device=self.device)
        self._no_torque = torch.zeros(self.num_envs, self._num_thrusters, 3, device=self.device)
        self._external_wrench = torch.zeros(self.num_envs, 1, 6, device=self.device)

        self._hydrostatics = Hydrostatics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.hydrostatics_cfg)

        self._hydrodynamics = Hydrodynamics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.hydrodynamics_cfg)

        # One actuator per thruster, so each gets its own force randomization
        self._thruster_dynamics = [
            PropellerActuator(num_envs=self.num_envs, device=self.device, dt=cfg.physics_dt, cfg=self.cfg.propeller_cfg)
            for _ in range(self._num_thrusters)
        ]

        # Buffers
        self.distance = torch.zeros(self.num_envs, device=self.device)
        self.previous_distance = torch.zeros(self.num_envs, device=self.device)
        self.distance_progress = torch.zeros(self.num_envs, device=self.device)
        self.initial_distance = torch.zeros(self.num_envs, device=self.device)

        self.bearing = torch.zeros(self.num_envs, device=self.device)
        self.previous_bearing = torch.zeros(self.num_envs, device=self.device)
        self.bearing_progress = torch.zeros(self.num_envs, device=self.device)
        self.initial_bearing = torch.zeros(self.num_envs, device=self.device)

        self.energy = torch.zeros(self.num_envs, device=self.device)
        self.desired_pos_b = torch.zeros(self.num_envs, 2, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Set the thruster targets; the actuators rate limit them every physics step
        for i, thruster in enumerate(self._thruster_dynamics):
            thruster.set_target_cmd(self._actions[:, i])

    def _apply_action(self):
        # Compute the hydrostatic and hydrodynamic forces
        robot_pos = self._robot.data.root_pos_w.clone()
        robot_quat = self._robot.data.root_quat_w.clone()
        robot_vel = self._robot.data.root_vel_w.clone()

        self._hydrostatic_force[:, 0, :] = self._hydrostatics.compute_archimedes_metacentric_local(
            robot_pos, robot_quat
        )
        drag = self._hydrodynamics.ComputeHydrodynamicsEffects(robot_quat, robot_vel)
        # Move the drag force from the CoM to the hydro point: add r x F, with r from the CoM to the hydro point
        lever = self._hydro_point_b - self._base_com_b
        self._hydrodynamic_force[:, 0, :3] = drag[:, :3]
        self._hydrodynamic_force[:, 0, 3:] = drag[:, 3:] + torch.linalg.cross(lever, drag[:, :3], dim=-1)
        combined = self._hydrostatic_force + self._hydrodynamic_force + self._external_wrench
        # The wrench composer wraps the inputs with wp.from_torch, so pass contiguous tensors
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=combined[..., :3].contiguous(), torques=combined[..., 3:].contiguous(), body_ids=self._base_link
        )

        # Update the thruster forces (zero thrust is written too, so no stale force stays in the composer)
        for i, thruster in enumerate(self._thruster_dynamics):
            self._thruster_forces[:, i] = thruster.update_forces()
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._thruster_forces, torques=self._no_torque, body_ids=self._thruster_ids
        )

    def _update_goal_frame(self):
        # Desired position in the robot frame (2D)
        self.desired_pos_b_3d, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )
        self.desired_pos_b[:, :2] = self.desired_pos_b_3d[:, :2]
        self.distance = torch.linalg.norm(self.desired_pos_b, dim=1)
        self.bearing = torch.atan2(self.desired_pos_b[:, 1], self.desired_pos_b[:, 0])

    def _get_observations(self) -> dict:
        # Update previous state
        self.previous_distance = self.distance.clone()
        self.previous_bearing = self.bearing.clone()

        self._update_goal_frame()

        obs = torch.cat(
            [
                self._actions,  # 3
                self._robot.data.root_lin_vel_b[:, :2],  # 2
                self._robot.data.root_ang_vel_b[:, 2].unsqueeze(1),  # 1
                torch.cos(self.bearing).unsqueeze(1),  # 1
                torch.sin(self.bearing).unsqueeze(1),  # 1
                self.distance.unsqueeze(1),  # 1
            ],
            dim=1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Distance progress - Potential field reward to guide policy convergence
        self.distance_progress = self.previous_distance - self.distance
        distance_progress_reward = self.distance_progress * self.cfg.distance_progress_reward_scale

        # Bearing progress - Potential field reward to guide policy convergence
        self.bearing_progress = torch.cos(self.bearing) - torch.cos(self.previous_bearing)
        bearing_progress_reward = self.bearing_progress * self.cfg.bearing_progress_reward_scale

        # Bearing reached - Reward for being closely aligned with the goal direction
        bearing_reached = 1.0 - torch.square(self.bearing / self.cfg.bearing_reached_threshold)  # 1 - (b/t)^2
        bearing_reached_reward = torch.clamp(bearing_reached, 0, 1) * self.cfg.bearing_reached_reward_scale

        # Reached goal
        goal_reward = torch.zeros(self.num_envs, device=self.device)
        goal_reward[self.distance < self.cfg.goal_reached_threshold] = self.cfg.goal_reached_scale

        # Energy - Smooth control input and minimizing unnecessary movements
        self.energy = torch.sum(torch.square(self._actions), dim=1)
        energy_norm = self.energy * self.step_dt / self.cfg.max_energy
        energy_reward = self.cfg.energy_penalty_scale * energy_norm

        # Velocity penalty - define linear velocity operating range
        root_lin_vel_b_x = self._robot.data.root_lin_vel_b[:, 0]
        # max_vel_comp = MAX(v_min -v, v - v_max, 0)
        max_vel_comp = torch.max(
            self.cfg.velocity_lower_bound - root_lin_vel_b_x, root_lin_vel_b_x - self.cfg.velocity_upper_bound
        )
        max_vel_comp = torch.clamp(max_vel_comp, min=0)
        velocity_penalty = torch.exp(self.cfg.velocity_penalty_coef * max_vel_comp) - 1
        velocity_penalty = self.cfg.velocity_penalty_scale * velocity_penalty

        # Time pressure
        time_reward = torch.ones(self.num_envs, device=self.device) * self.cfg.time_penalty_scale

        rewards = {
            "1_distance_progress": distance_progress_reward,
            "2_bearing_progress": bearing_progress_reward,
            "3_goal_reached": goal_reward,
            "4_bearing_reached": bearing_reached_reward,
            "5_energy": energy_reward,
            "6_velocity": velocity_penalty,
            "7_time": time_reward,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self._update_goal_frame()

        # Finish episode if the goal is reached
        done = torch.zeros_like(time_out)
        done[self.distance < self.cfg.goal_reached_threshold] = True
        return done, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = self.distance[env_ids].mean()
        final_bearing_to_goal = self.bearing[env_ids].mean()
        final_energy = self.energy[env_ids].mean()
        final_velocity = self._robot.data.root_lin_vel_b[env_ids, 0].mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/done"] = torch.count_nonzero(self.reset_terminated[env_ids]).item() / len(env_ids)
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item() / len(
            env_ids
        )
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/final_bearing_to_goal"] = final_bearing_to_goal.item()
        extras["Metrics/final_energy"] = final_energy.item()
        extras["Metrics/final_velocity"] = final_velocity.item()
        self.extras["log"].update(extras)

        for thruster in self._thruster_dynamics:
            thruster.reset(env_ids)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new goal position
        self.initial_bearing[env_ids] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(
            self.cfg.min_target_bearing, self.cfg.max_target_bearing
        )
        self.bearing[env_ids] = self.initial_bearing[env_ids]
        self.previous_bearing[env_ids] = self.initial_bearing[env_ids]

        self.initial_distance[env_ids] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(
            self.cfg.min_target_distance, self.cfg.max_target_distance
        )
        self.distance[env_ids] = self.initial_distance[env_ids]
        self.previous_distance[env_ids] = self.initial_distance[env_ids]
        self._desired_pos_w[env_ids, 0] = torch.cos(self.initial_bearing[env_ids]) * self.initial_distance[env_ids]
        self._desired_pos_w[env_ids, 1] = torch.sin(self.initial_bearing[env_ids]) * self.initial_distance[env_ids]
        self._desired_pos_w[env_ids, 2] = 0.0  # only in 2D
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Add random initial velocity to vx and rz
        if self.cfg.enable_v0_randomizations:
            default_root_state[:, 7] = torch.rand(len(env_ids), device=self.device) * 0.7
            default_root_state[:, 12] = torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0

        # Random lateral CoM offset of base_link, from the default CoM (so it doesn't accumulate over resets)
        if self.cfg.enable_com_randomization:
            ids_cpu = env_ids.to("cpu")
            com_offset = torch.rand(len(env_ids)) * 0.1 - 0.05
            com = self._default_coms.clone()
            com[ids_cpu, self._base_link[0], 1] += com_offset
            self._robot.root_physx_view.set_coms(com, ids_cpu)
            self._base_com_b[env_ids] = com[ids_cpu, self._base_link[0], :3].to(self.device)

        # Randomize external wrench
        if self.cfg.enable_external_wrench_randomization:
            self._external_wrench[env_ids, 0, 0] = torch.rand(len(env_ids), device=self.device) * 0.6 - 0.3  # x force
            self._external_wrench[env_ids, 0, 1] = torch.rand(len(env_ids), device=self.device) * 0.6 - 0.3  # y force
            self._external_wrench[env_ids, 0, 5] = torch.rand(len(env_ids), device=self.device) * 0.6 - 0.3  # r force

        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_link_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.5)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
