# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the IADYS Jellyfishbot (Expert) ASV."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

JELLYFISHBOT_ASSET_DIR = os.environ.get("JELLYFISHBOT_ASSET_DIR", os.path.expanduser("~/assets/jellyfishbot"))
"""Local folder holding the flattened Jellyfishbot USD (not tracked in git). Override with ``JELLYFISHBOT_ASSET_DIR``.

The USD is generated from ``jfb_description/urdf/jfb.urdf.xacro`` (not the Gazebo top level), converted with
``convert_urdf.py --joint-target-type none`` and post-processed with ``scripts/tools/kingfisher/flatten_usd.py``.
"""

##
# Configuration
##

JELLYFISHBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{JELLYFISHBOT_ASSET_DIR}/jellyfishbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=10000.0,
            max_angular_velocity=10000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        # The water surface is at z=0, the same height as the ground plane, and the floaters and thrusters reach
        # z~-0.19 when floating. Buoyancy and drag are computed analytically, not from contacts.
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base_link rests about 0.20 m above the water (as in jfb_gazebo)
        pos=(0.0, 0.0, 0.20),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={
        # The three propeller joints are passive; thrust is applied as an external force
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Jellyfishbot."""
