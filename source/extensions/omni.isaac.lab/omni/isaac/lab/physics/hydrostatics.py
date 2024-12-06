# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from omni.isaac.lab.utils.math import euler_xyz_from_quat, matrix_from_quat

"""
Following Fossen's Equation,
Fossen, T. I. (1991). Nonlinear modeling and control of Underwater Vehicles. Doctoral thesis, Department of Engineering Cybernetics, Norwegian Institute of Technology (NTH), June 1991.
"""


class Hydrostatics:
    def __init__(self, num_envs, device, gravity, params):
        self._num_envs = num_envs
        self.device = device
        self.drag = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self.device)

        # data
        self.average_hydrostatics_force_value = params["average_hydrostatics_force_value"]
        self.amplify_torque = params["amplify_torque"]
        self.metacentric_width = params["box_width"] / 2
        self.metacentric_length = params["box_length"] / 2
        self.waterplane_area = params["waterplane_area"]
        self.water_density = params["water_density"]
        self.max_draught = params["max_draught"]
        self.draught_offset = params["draught_offset"]

        # Buoyancy
        self.gravity = gravity

        self.archimedes_force_global = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self.device)
        self.archimedes_torque_global = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self.device)
        self.archimedes_force_local = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self.device)
        self.archimedes_torque_local = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self.device)

        return

    def compute_archimedes_metacentric_global(self, submerged_volume, rpy):
        roll, pitch = rpy[:, 0], rpy[:, 1]  # roll and pich are given in global frame

        # compute buoyancy force
        self.archimedes_force_global[:, 2] = -self.water_density * self.gravity * submerged_volume

        # torques expressed in global frame, size is (num_envs,3)
        self.archimedes_torque_global[:, 0] = (
            -1 * self.metacentric_width * (torch.sin(roll) * self.archimedes_force_global[:, 2])
        )
        self.archimedes_torque_global[:, 1] = (
            -1 * self.metacentric_length * (torch.sin(pitch) * self.archimedes_force_global[:, 2])
        )

        self.archimedes_torque_global[:, 0] = (
            -1 * self.metacentric_width * (torch.sin(roll) * self.average_hydrostatics_force_value)
        )  # cannot multiply by the hydrostatics force in isaac sim because of the simulation rate (low then high value)
        self.archimedes_torque_global[:, 1] = (
            -1 * self.metacentric_length * (torch.sin(pitch) * self.average_hydrostatics_force_value)
        )

        # debugging
        # print("self.archimedes_force global: ", self.archimedes_force_global[0,:])
        # print("self.archimedes_torque global: ", self.archimedes_torque_global[0,:])

        return self.archimedes_force_global, self.archimedes_torque_global

    def compute_submerged_volume(self, position):

        draught = torch.clamp(self.draught_offset - position[:, 2], 0, self.max_draught)

        submerged_volume = draught * self.waterplane_area

        return submerged_volume

    def compute_archimedes_metacentric_local(self, position, quaternions):

        roll, pitch, yaw = euler_xyz_from_quat(quaternions)
        euler = torch.stack((roll, pitch, yaw), dim=1)

        # get archimedes global force
        submerged_volume = self.compute_submerged_volume(position)
        self.compute_archimedes_metacentric_global(submerged_volume, euler)

        # get rotation matrix from quaternions in world frame, size is (3*num_envs, 3)
        R = matrix_from_quat(quaternions)

        # Arobot = Rworld * Aworld. Resulting matrix should be size (3*num_envs, 3) * (num_envs,3) =(num_envs,3)
        self.archimedes_force_local = torch.bmm(
            R.mT, torch.unsqueeze(self.archimedes_force_global, 1).mT
        )  # add batch dimension to tensor and transpose it
        self.archimedes_force_local = self.archimedes_force_local.mT.squeeze(1)  # remove batch dimension to tensor

        self.archimedes_torque_local = torch.bmm(R.mT, torch.unsqueeze(self.archimedes_torque_global, 1).mT)
        self.archimedes_torque_local = self.archimedes_torque_local.mT.squeeze(1)

        # not sure if torque have to be multiply by the rotation matrix also.
        self.archimedes_torque_local = self.archimedes_torque_global

        # print(f"archimedes_force_global: {self.archimedes_force_global[0,:]}")
        # print(f"archimedes_force_local: {self.archimedes_force_local[0,:]}")

        return torch.hstack([
            self.archimedes_force_local,
            self.archimedes_torque_local * self.amplify_torque,
        ])
