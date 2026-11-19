# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the Jellyfishbot's steady-state dynamics in Isaac Lab with its gz-sim model.

Each environment holds one fixed thruster command (left, right, lateral) with every randomization off. After the run,
the script prints the mean of the last seconds next to the values from ``jellyfishbot_simulation/README.md``.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/jellyfishbot/check_dynamics.py --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the Jellyfishbot dynamics against its Gazebo model.")
parser.add_argument("--duration", type=float, default=20.0, help="Seconds each command is held.")
parser.add_argument("--average", type=float, default=2.0, help="Seconds at the end that are averaged.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from isaaclab_tasks.direct.jellyfishbot.jellyfishbot_env import JellyfishbotEnv, JellyfishbotEnvCfg

# name, (left, right, lateral), the quantity compared, its Gazebo value
CASES = [
    ("idle", (0.0, 0.0, 0.0), "z", 0.20),
    ("full forward", (1.0, 1.0, 0.0), "vx", 1.02),
    ("full reverse", (-1.0, -1.0, 0.0), "vx", -0.89),
    ("full lateral (port)", (0.0, 0.0, 1.0), "vy", 0.56),
    ("spin left -1 / right +1", (-1.0, 1.0, 0.0), "wz", 1.97),
    ("spin left -0.5 / right +0.5", (-0.5, 0.5, 0.0), "wz", 1.1),
]
"""Gazebo values from the jellyfishbot_simulation README. At full forward Gazebo also shows 0.06 rad bow-up."""


def main():
    cfg = JellyfishbotEnvCfg()
    cfg.scene.num_envs = len(CASES)
    cfg.sim.device = args_cli.device
    cfg.debug_vis = False
    cfg.episode_length_s = 10.0 * args_cli.duration
    # Goals far away, so no env reaches one and resets
    cfg.min_target_distance = 1.0e4
    cfg.max_target_distance = 1.0e4
    # No randomization
    cfg.enable_v0_randomizations = False
    cfg.enable_com_randomization = False
    cfg.enable_external_wrench_randomization = False
    cfg.propeller_cfg.enable_randomization = False
    cfg.propeller_cfg.enable_init_randomization = False
    cfg.hydrodynamics_cfg.use_drag_randomization = False

    env = JellyfishbotEnv(cfg)
    env.reset()
    # The first reset spreads the episode lengths at random; restart them so no env times out
    env.episode_length_buf.zero_()

    actions = torch.tensor([case[1] for case in CASES], device=env.device)
    num_steps = int(args_cli.duration / env.step_dt)
    num_avg = max(1, int(args_cli.average / env.step_dt))
    sums = {k: torch.zeros(env.num_envs, device=env.device) for k in ("z", "vx", "vy", "wz", "roll", "pitch")}

    with torch.inference_mode():
        for step in range(num_steps):
            env.step(actions)
            if step >= num_steps - num_avg:
                data = env._robot.data
                roll, pitch, _ = euler_xyz_from_quat(data.root_quat_w)
                sums["z"] += data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
                sums["vx"] += data.root_lin_vel_b[:, 0]
                sums["vy"] += data.root_lin_vel_b[:, 1]
                sums["wz"] += data.root_ang_vel_b[:, 2]
                sums["roll"] += wrap_to_pi(roll)
                sums["pitch"] += wrap_to_pi(pitch)

    mean = {k: (v / num_avg).cpu() for k, v in sums.items()}
    print(f"\nTotal mass: {env._robot_mass.item():.3f} kg ({env._robot_weight:.1f} N)")
    print(f"Mean of the last {args_cli.average:.1f} s of {args_cli.duration:.1f} s (pitch < 0 is bow-up)\n")
    columns = " ".join(f"{k:>7}" for k in ("z", "vx", "vy", "wz", "roll", "pitch"))
    header = f"{'case':<28} {columns}   {'check':<6} {'isaac':>7} {'gazebo':>7} {'err %':>6}"
    print(header)
    print("-" * len(header))
    for i, (name, _, key, ref) in enumerate(CASES):
        val = mean[key][i].item()
        err = 100.0 * (val - ref) / abs(ref)
        row = " ".join(f"{mean[k][i].item():7.3f}" for k in ("z", "vx", "vy", "wz", "roll", "pitch"))
        print(f"{name:<28} {row}   {key:<6} {val:7.3f} {ref:7.3f} {err:6.1f}")
    print("\nGazebo at full forward: pitch about -0.06 rad (0.06 rad bow-up).\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
