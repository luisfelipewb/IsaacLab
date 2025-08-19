# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab_assets.robots.kingfisher import KINGFISHER_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class KingfisherEnvWindow(BaseEnvWindow):
    """Window manager for the Kingfisher environment."""

    def __init__(self, env: KingfisherEnv, window_name: str = "IsaacLab"):
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
class KingfisherEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 60.0
    physics_dt = 1 / 50.0  # 50 Hz
    decimation = 5
    step_dt = physics_dt * decimation  # 10 Hz
    action_space = 2
    observation_space = 8
    state_space = 0
    debug_vis = True

    ui_window_class_type = KingfisherEnvWindow

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
    robot: ArticulationCfg = KINGFISHER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Hydrostatics
    hydrostatics_cfg: HydrostaticsCfg = HydrostaticsCfg(
        mass=35.0,  # Kg considering added sensors
        width=1.0,  # Kingfisher/Heron width 1.0m in Spec Sheet
        length=1.3,  # Kingfisher/Heron length 1.3m in Spec Sheet
        waterplane_area=0.33,  # 0.15 width * 1.1 length * 2 hulls
        draught_offset=0.21986,  # Distance from base_link to bottom of the hull
        max_draught=0.20,  # Kingfisher/Heron draught 120mm in Spec Sheet
        average_hydrostatics_force=275.0,
    )

    # Hydrdynamics
    # Nominal
    # linear    [16.44998712, 15.79776044, 100, 13, 13, 6]
    # quadratic [2.942, 2.7617212, 10, 5, 5, 5]
    # SID
    # linear    [0.0, 99.99, 99.99, 13.0, 13.0, 0.82985084]
    # quadratic [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    hydrodynamics_cfg: HydrodynamicsCfg = HydrodynamicsCfg(
        linear_damping=[0.0, 99.99, 99.99, 13.0, 13.0, 5.83],
        quadratic_damping=[17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724],
        use_drag_randomization=False,
        linear_damping_rand=[0.1, 0.1, 0.0, 0.0, 0.0, 0.1],
        quadratic_damping_rand=[0.1, 0.1, 0.0, 0.0, 0.0, 0.1],
    )

    # Thruster dynamics
    propeller_cfg: PropellerActuatorCfg = PropellerActuatorCfg(
        cmd_lower_range=-1.0,
        cmd_upper_range=1.0,
        command_rate=1.0,
        forces=[
            -4.0,  # -1.0
            -4.0,  # -0.9
            -4.0,  # -0.8
            -4.0,  # -0.7
            -2.0,  # -0.6
            -1.0,  # -0.5
            0.0,  # -0.4
            0.0,  # -0.3
            0.0,  # -0.2
            0.0,  # -0.1
            0.0,  # 0.0
            0.0,  # 0.1
            0.0,  # 0.2
            0.5,  # 0.3
            1.5,  # 0.4
            4.75,  # 0.5
            8.25,  # 0.6
            16.0,  # 0.7
            19.5,  # 0.8
            19.5,  # 0.9
            19.5,  # 1.0
        ],
        interp_resolution=1001,
        enable_randomization=True,
        randomization_range=0.1,
        enable_init_randomization=True,
    )

    max_energy = 2.0  # Max of 1.0 per thruster

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
    velocity_penalty_scale = 1.0

    # Reward coeficients
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


class KingfisherEnv(DirectRLEnv):
    cfg: KingfisherEnvCfg

    def __init__(self, cfg: KingfisherEnvCfg, render_mode: str | None = None, **kwargs):
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
        self._left_thruster_id = self._robot.find_bodies("thruster_left")[0]
        self._right_thruster_id = self._robot.find_bodies("thruster_right")[0]

        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Forces
        self._hydrodynamic_force = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self._hydrostatic_force = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self._thruster_forces_left = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thruster_forces_right = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._no_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._external_wrench = torch.zeros(self.num_envs, 1, 6, device=self.device)

        self._hydrostatics = Hydrostatics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.hydrostatics_cfg)

        self._hydrodynamics = Hydrodynamics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.hydrodynamics_cfg)

        self._thruster_dynamics_left = PropellerActuator(
            num_envs=self.num_envs, device=self.device, dt=cfg.physics_dt, cfg=self.cfg.propeller_cfg
        )
        self._thruster_dynamics_right = PropellerActuator(
            num_envs=self.num_envs, device=self.device, dt=cfg.physics_dt, cfg=self.cfg.propeller_cfg
        )

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
        # Override the actions for debugging
        # self._actions[:,0] = 0.6
        # self._actions[:,1] = 0.6

        # Compute the thruster forces based on the actions.
        self._thruster_dynamics_left.set_target_cmd(self._actions[:, 0])
        self._thruster_dynamics_right.set_target_cmd(self._actions[:, 1])

    def _apply_action(self):
        # Compute the hydrostatic and hydrodynamic forces
        robot_pos = self._robot.data.root_pos_w.clone()
        robot_quat = self._robot.data.root_quat_w.clone()
        robot_vel = self._robot.data.root_vel_w.clone()

        self._hydrostatic_force[:, 0, :] = self._hydrostatics.compute_archimedes_metacentric_local(
            robot_pos, robot_quat
        )
        self._hydrodynamic_force[:, 0, :] = self._hydrodynamics.ComputeHydrodynamicsEffects(robot_quat, robot_vel)
        combined = self._hydrostatic_force + self._hydrodynamic_force + self._external_wrench
        self._robot.set_external_force_and_torque(combined[..., :3], combined[..., 3:], body_ids=self._base_link)

        # Update the thruster forces
        self._thruster_forces_left[:, 0] = self._thruster_dynamics_left.update_forces()
        self._thruster_forces_right[:, 0] = self._thruster_dynamics_right.update_forces()
        if self._thruster_forces_left.any():
            self._robot.set_external_force_and_torque(
                self._thruster_forces_left, self._no_torque, body_ids=self._left_thruster_id
            )
        if self._thruster_forces_right.any():
            self._robot.set_external_force_and_torque(
                self._thruster_forces_right, self._no_torque, body_ids=self._right_thruster_id
            )

    def _get_observations(self) -> dict:

        # Update previous state
        self.previous_distance = self.distance.clone()
        self.previous_bearing = self.bearing.clone()

        # Desired position in the robot frame (2D)
        self.desired_pos_b_3d, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )
        self.desired_pos_b[:, :2] = self.desired_pos_b_3d[:, :2]
        self.distance = torch.linalg.norm(self.desired_pos_b, dim=1)
        self.bearing = torch.atan2(self.desired_pos_b[:, 1], self.desired_pos_b[:, 0])

        obs = torch.cat(
            [
                self._actions,  # 2
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

        # Distance progress - Potentail field reward to guide policy convergence
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

        # Velocity penalty - define linear velocity operating rage
        root_lin_vel_b_x = self._robot.data.root_lin_vel_b[:, 0]
        velocity_penalty = torch.zeros(self.num_envs, device=self.device)
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

        # Desired position in the robot frame (2D)
        self.desired_pos_b_3d, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )
        self.desired_pos_b[:, :2] = self.desired_pos_b_3d[:, :2]
        self.distance = torch.linalg.norm(self.desired_pos_b, dim=1)
        self.bearing = torch.atan2(self.desired_pos_b[:, 1], self.desired_pos_b[:, 0])

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
        final_velocity = self._robot.data.root_lin_vel_b[:, 0].mean()
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

        self._thruster_dynamics_left.reset(env_ids)
        self._thruster_dynamics_right.reset(env_ids)

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

        # Generate random CoM offset (2d)
        if self.cfg.enable_com_randomization:
            com_offset = torch.rand(len(env_ids), device=self.device) * 0.1 - 0.05
            com = self._robot.root_physx_view.get_coms().to(self.device)
            com[env_ids, 0, 1] += com_offset
            if env_ids is None:
                ids_cpu = self._ALL_INDICES_CPU
            else:
                ids_cpu = env_ids.to("cpu")
            self._robot.root_physx_view.set_coms(com.to("cpu"), ids_cpu)

        # Randomize external wrench
        if self.cfg.enable_external_wrench_randomization:
            self._external_wrench[env_ids, 0, 0] = torch.rand(len(env_ids), device=self.device) * 0.6 - 0.3  # x force
            self._external_wrench[env_ids, 0, 1] = torch.rand(len(env_ids), device=self.device) * 0.6 - 0.3  # y force
            self._external_wrench[env_ids, 0, 5] = torch.rand(len(env_ids), device=self.device) * 0.6 - 0.3  # r force

        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_link_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the_robot_mass first tome
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
