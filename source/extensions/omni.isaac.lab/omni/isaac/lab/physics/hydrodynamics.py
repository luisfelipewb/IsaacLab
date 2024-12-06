# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_rotate_inverse


@configclass
class HydrodynamicsCfg:
    """Configuration for hydrodynamics."""

    # Add drag disturbances
    use_drag_randomization: bool = False
    # Proportion of drag randomization for each drag coefficient
    # If it is 0.1 it means 0.9 to 1.1
    # Linear
    u_linear_rand: float = 0.1  # Forward
    v_linear_rand: float = 0.1  # Lateral
    w_linear_rand: float = 0.0  # Vertical. In 2D, neglectable
    p_linear_rand: float = 0.0  # Roll. In 2D, neglectable
    q_linear_rand: float = 0.0  # Pitch. In 2D, neglectable
    r_linear_rand: float = 0.1  # Yaw
    # Quadratic
    u_quad_rand: float = 0.1  # Forward
    v_quad_rand: float = 0.1  # Lateral
    w_quad_rand: float = 0.0  # Vertical. In 2D, neglectable
    p_quad_rand: float = 0.0  # Roll. In 2D, neglectable
    q_quad_rand: float = 0.0  # Pitch. In 2D, neglectable
    r_quad_rand: float = 0.1  # Yaw

    linear_damping: list = [0.0, 99.99, 99.99, 13.0, 13.0, 5.83]
    # Nominal [16.44998712, 15.79776044, 100, 13, 13, 6]
    # SID [0.0, 99.99, 99.99, 13.0, 13.0, 0.82985084]
    quadratic_damping: list = [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    # Nominal [2.942, 2.7617212, 10, 5, 5, 5]
    # SID [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    linear_damping_forward_speed: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    offset_linear_damping: float = 0.0
    offset_lin_forward_damping_speed: float = 0.0
    offset_nonlin_damping: float = 0.0
    scaling_damping: float = 1.0
    offset_added_mass: float = 0.0


class Hydrodynamics:
    def __init__(self, num_envs, device, cfg: HydrodynamicsCfg):

        self.cfg = cfg

        # new variable can be randomized
        self.linear_damping = self.cfg.linear_damping.copy()
        self.quadratic_damping = self.cfg.quadratic_damping.copy()

        # linear_rand range, calculated as a percentage of the base damping coefficients
        self._linear_rand = torch.tensor(
            [
                self.cfg.u_linear_rand * self.linear_damping[0],
                self.cfg.v_linear_rand * self.linear_damping[1],
                self.cfg.w_linear_rand * self.linear_damping[2],
                self.cfg.p_linear_rand * self.linear_damping[3],
                self.cfg.q_linear_rand * self.linear_damping[4],
                self.cfg.r_linear_rand * self.linear_damping[5],
            ],
            device=device,
        )
        self._quad_rand = torch.tensor(
            [
                self.cfg.u_quad_rand * self.quadratic_damping[0],
                self.cfg.v_quad_rand * self.quadratic_damping[1],
                self.cfg.w_quad_rand * self.quadratic_damping[2],
                self.cfg.p_quad_rand * self.quadratic_damping[3],
                self.cfg.q_quad_rand * self.quadratic_damping[4],
                self.cfg.r_quad_rand * self.quadratic_damping[5],
            ],
            device=device,
        )

        self._num_envs = num_envs
        self.device = device
        self.drag = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self.device)

        # damping parameters (individual set for each environment)
        self.linear_damping = torch.tensor([self.linear_damping] * num_envs, device=self.device)  # num_envs * 6
        self.quadratic_damping = torch.tensor([self.quadratic_damping] * num_envs, device=self.device)  # num_envs * 6
        self.linear_damping_forward_speed = torch.tensor(self.cfg.linear_damping_forward_speed, device=self.device)
        # damping parameters randomization
        if self.cfg.use_drag_randomization:
            # Applying uniform noise as an example
            self.linear_damping += (torch.rand_like(self.linear_damping) * 2 - 1) * self._linear_rand
            self.quadratic_damping += (torch.rand_like(self.quadratic_damping) * 2 - 1) * self._quad_rand
        # Debug : print the initialized coefficients
        # print("linear_damping: ", self.linear_damping)

        # coriolis
        self._Ca = torch.zeros([6, 6], device=self.device)
        self.added_mass = torch.zeros([num_envs, 6], device=self.device)

        # acceleration
        self._filtered_acc = torch.zeros([6], device=self.device)
        self._last_vel_rel = torch.zeros([6], device=self.device)

        return

    def reset_coefficients(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Resets the drag coefficients for the specified environments.
        Args:
            env_ids (torch.Tensor): Indices of the environments to reset.
        """
        if self.cfg.use_drag_randomization:
            # Generate random noise
            noise_linear = (torch.rand((len(env_ids), 6), device=self.device) * 2 - 1) * self._linear_rand
            noise_quad = (torch.rand((len(env_ids), 6), device=self.device) * 2 - 1) * self._quad_rand

            # Apply noise to the linear and quadratic damping coefficients
            # Use indexing to update only the specified environments
            self.linear_damping[env_ids] = (
                torch.tensor([self.cfg.linear_damping], device=self.device).expand_as(noise_linear) + noise_linear
            )
            self.quadratic_damping[env_ids] = (
                torch.tensor([self.cfg.quadratic_damping], device=self.device).expand_as(noise_quad) + noise_quad
            )
        return

    def ComputeDampingMatrix(self, vel):
        """
        // From Antonelli 2014: the viscosity of the fluid causes
        // the presence of dissipative drag and lift forces on the
        // body. A common simplification is to consider only linear
        // and quadratic damping terms and group these terms in a
        // matrix Drb
        """
        # print("vel: ", vel)
        lin_damp = (
            self.linear_damping
            + self.cfg.offset_linear_damping
            - (self.linear_damping_forward_speed + self.cfg.offset_lin_forward_damping_speed)
        )
        # print("lin_damp: ", lin_damp)
        quad_damp = ((self.quadratic_damping + self.cfg.offset_nonlin_damping).mT * torch.abs(vel.mT)).mT
        # print("quad_damp: ", quad_damp)
        # scaling and adding both matrices
        damping_matrix = (lin_damp + quad_damp) * self.cfg.scaling_damping
        # print("damping_matrix: ", damping_matrix)
        return damping_matrix

    def ComputeHydrodynamicsEffects(self, quaternions, world_vel):

        self.local_lin_velocities = quat_rotate_inverse(quaternions, world_vel[:, :3])
        self.local_ang_velocities = quat_rotate_inverse(quaternions, world_vel[:, 3:])
        self.local_velocities = torch.hstack([self.local_lin_velocities, self.local_ang_velocities])

        # Update damping matrix
        damping_matrix = self.ComputeDampingMatrix(self.local_velocities)

        # Damping forces and torques
        self.drag = -1 * damping_matrix * self.local_velocities

        return self.drag
