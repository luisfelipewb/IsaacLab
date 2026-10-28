# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Flatten a Kingfisher USD produced by ``convert_urdf.py`` into one self-contained file and fix it for simulation.

The importer's top-level USD is a stub that loads ``configuration/*.usd`` as payloads. This script bakes them into a
single file and then:

* makes the ``collisions`` prims non-instanceable. Isaac Lab cannot change properties on instanced prims, so without
  this ``UsdFileCfg.collision_props`` (e.g. ``collision_enabled=False``) has no effect on the hull;
* gives the massless links (sensor frames, sail motor) a negligible explicit mass. A link with mass 0 and no
  collider gets the PhysX default of 1 kg, which added 10 kg to the boat, mostly high above the deck.

It only needs ``pxr``, not Kit, so run it with Isaac Sim's Python and its bundled USD libraries:

.. code-block:: bash

    L=$(ls -d ~/isaacsim/extscache/omni.usd.libs-*)
    env -u PYTHONPATH PYTHONPATH=$L LD_LIBRARY_PATH=$L/bin ~/isaacsim/kit/python/bin/python3 \\
        scripts/tools/kingfisher/flatten_usd.py ~/kf-usd/kingfisher.usd ~/assets/kingfisher/kingfisher.usd

"""

import argparse

from pxr import Gf, Usd, UsdPhysics

MASSLESS_LINK_MASS = 1.0e-3
"""Mass [kg] given to links that have no mass in the URDF."""


def main():
    parser = argparse.ArgumentParser(description="Flatten and fix a Kingfisher USD from the URDF importer.")
    parser.add_argument("input", help="Top-level USD written by convert_urdf.py.")
    parser.add_argument("output", help="Path of the flattened USD.")
    args = parser.parse_args()

    stage = Usd.Stage.Open(Usd.Stage.Open(args.input, Usd.Stage.LoadAll).Flatten())

    for prim in stage.Traverse():
        if prim.IsInstance() and prim.GetName() == "collisions":
            prim.SetInstanceable(False)
            print(f"[INFO] De-instanced: {prim.GetPath()}")

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        if mass_api.GetMassAttr().Get() > 0.0:
            continue
        mass_api.GetMassAttr().Set(MASSLESS_LINK_MASS)
        mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        print(f"[INFO] Set mass {MASSLESS_LINK_MASS} kg: {prim.GetPath()}")

    stage.GetRootLayer().Export(args.output)
    print(f"[INFO] Wrote: {args.output}")


if __name__ == "__main__":
    main()
