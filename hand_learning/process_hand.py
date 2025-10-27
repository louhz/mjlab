#!/usr/bin/env python3
"""
Load an MJCF .xml with MuJoCo Warp (GPU), step it, and display with MuJoCo's viewer.

Usage:
  python run_mjwarp_viewer.py --xml /path/to/scene.xml --steps 100000 --nworld 1 --realtime
"""

import argparse
import time
import numpy as np
import mujoco
from mujoco import viewer
import mujoco_warp as mjw
import warp as wp


# imitation learning witha adapative force and gesture control

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True, help="Path to MJCF .xml file")
    parser.add_argument("--steps", type=int, default=10_0000, help="Max simulation steps")
    parser.add_argument("--nworld", type=int, default=1, help="Parallel worlds on GPU")
    parser.add_argument("--realtime", action="store_true",
                        help="Sleep to ~real-time using model.opt.timestep")
    args = parser.parse_args()

    # --- Load CPU model/data for rendering ---
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # --- Upload to GPU (MJWarp) ---
    m_gpu = mjw.put_model(model)
    d_gpu = mjw.make_data(model, nworld=args.nworld)

    # Initialize GPU state from CPU (optionally randomize)
    if args.nworld > 1:
        qpos = np.tile(data.qpos, (args.nworld, 1))
        qvel = np.tile(data.qvel, (args.nworld, 1))
    else:
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
    wp.copy(d_gpu.qpos, wp.array(qpos, dtype=float))
    wp.copy(d_gpu.qvel, wp.array(qvel, dtype=float))

    # --- Launch passive viewer (non-blocking) ---
    # We advance physics ourselves and call viewer.sync() each frame.
    with viewer.launch_passive(model, data) as v:  # requires mujoco>=2.3.3
        step = 0
        while v.is_running() and step < args.steps:
            # Step physics on GPU
            mjw.step(m_gpu, d_gpu)

            # Copy world 0 back to CPU for visualization
            qpos_host = d_gpu.qpos.numpy()
            if args.nworld > 1:
                data.qpos[:] = qpos_host[0]
            else:
                data.qpos[:] = qpos_host
            mujoco.mj_forward(model, data)

            # Draw this frame
            v.sync()

            # Optional real-time pacing
            if args.realtime:
                time.sleep(model.opt.timestep)

            step += 1

    print(f"[MJWarp] Finished at step {step} (viewer closed or step limit reached).")


if __name__ == "__main__":
    wp.init()  # init Warp runtime
    main()
