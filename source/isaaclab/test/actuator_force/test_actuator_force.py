# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from omni.isaac.lab.actuator_force.actuator_force import PropellerActuator, PropellerActuatorCfg


def custom_print(input_tensor):
    "Print with character alignment"
    for i in range(input_tensor.shape[0]):
        print(f"{i} l:{input_tensor[i, 0]:+07.3f}, r:{input_tensor[i, 1]:+07.3f}", end=" | ")
    print()


def test_new(
    iters=10, device="cpu", num_envs=2, physics_dt=1.0 / 50.0, thrust_cmds: torch.Tensor = None, command_rate=1.0
):

    propeller_cfg = PropellerActuatorCfg(
        cmd_lower_range=-1.0,
        cmd_upper_range=1.0,
        command_rate=command_rate,
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
            0.0,  # 0.2thurster_dynamics_left
            0.5,  # 0.3
            1.5,  # 0.4
            4.75,  # 0.5
            8.25,  # 0.6
            16.0,  # 0.7
            19.5,  # 0.8
            19.5,  # 0.9
            19.5,  # 1.0
        ],
    )
    thurster_dynamics_left = PropellerActuator(num_envs=num_envs, device=device, dt=physics_dt, cfg=propeller_cfg)

    thurster_dynamics_right = PropellerActuator(num_envs=num_envs, device=device, dt=physics_dt, cfg=propeller_cfg)

    thurster_dynamics_left.set_target_cmd(thrust_cmds[..., 0])
    thurster_dynamics_right.set_target_cmd(thrust_cmds[..., 1])

    for i in range(iters):
        thurster_dynamics_left.update_forces()
        thurster_dynamics_right.update_forces()

        # combine left and right forces
        forces = torch.stack([thurster_dynamics_left.get_forces(), thurster_dynamics_right.get_forces()], dim=-1)
        print(f"Step {i+1:02d}", end=" | ")
        custom_print(forces)

    print("reset")
    thurster_dynamics_left.reset()
    thurster_dynamics_right.reset()

    for i in range(5):
        thurster_dynamics_left.update_forces()
        thurster_dynamics_right.update_forces()

        # combine left and right forces
        forces = torch.stack([thurster_dynamics_left.get_forces(), thurster_dynamics_right.get_forces()], dim=-1)
        print(f"Step {i+1:02d}", end=" | ")
        custom_print(forces)

    # set target commands again
    print("set target commands again")
    thurster_dynamics_left.set_target_cmd(thrust_cmds[..., 0])
    thurster_dynamics_right.set_target_cmd(thrust_cmds[..., 1])

    for i in range(iters):
        thurster_dynamics_left.update_forces()
        thurster_dynamics_right.update_forces()
        forces = torch.stack([thurster_dynamics_left.get_forces(), thurster_dynamics_right.get_forces()], dim=-1)
        print(f"Step {i+1:02d}", end=" | ")
        custom_print(forces)


if __name__ == "__main__":

    device = "cuda:0"
    device = "cpu"
    command_rate = 1.0
    physics_dt = 1.0 / 50.0  # 50Hz
    num_envs = 6
    thrust_cmds = torch.tensor(
        [[0.0, -0.0], [0.2, -0.2], [0.4, -0.4], [0.6, -0.6], [0.8, -0.8], [1.0, -1.0]],
        dtype=torch.float32,
        device=device,
    )

    iters = 50

    print("===================================== Input commands")
    custom_print(thrust_cmds)
    print("===================================== NEW - split")
    test_new(
        iters=iters,
        device=device,
        num_envs=num_envs,
        physics_dt=physics_dt,
        thrust_cmds=thrust_cmds,
        command_rate=command_rate,
    )
