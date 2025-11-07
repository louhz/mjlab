# -*- coding: utf-8 -*-
"""
Synchronized multi-arm playback in MuJoCo Warp (GPU) with per-group start delays,
time compression, robust DoF/limit alignment, and 0–255→angle hand mapping.

Notes
- Physics is stepped on GPU with MuJoCo Warp (MJWarp).
- PD torques are computed on GPU via a small Warp kernel that writes into d.qfrc_applied.
- Viewer is optional (CPU viewer mirrors GPU state); offscreen recording is supported.
"""

import os
import ast
from pathlib import Path
import numpy as np

import mujoco
import mujoco_warp as mjw      # MJWarp (GPU stepping)
import warp as wp              # NVIDIA Warp (kernels / device arrays)

try:
    import imageio.v2 as imageio   # for mp4 recording
except Exception:
    imageio = None

# ------------------------------------------------------------
# Group parameters (names, per-DoF kp, and joint limits in radians)
# ------------------------------------------------------------
NOVA2_PARAMS = [
    {"joint": "nova2joint1", "kp": 300.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint2", "kp": 300.0, "ctrlrange": (-3.14,  3.14)},
    {"joint": "nova2joint3", "kp": 300.0, "ctrlrange": (-2.79,  2.79)},
    {"joint": "nova2joint4", "kp": 250.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint5", "kp": 200.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint6", "kp": 1500.0, "ctrlrange": (-6.28,  6.28)},
]
NOVA5_PARAMS = [
    {"joint": "nova5joint1", "kp": 300.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint2", "kp": 3000.0, "ctrlrange": (-3.14,  3.14)},
    {"joint": "nova5joint3", "kp": 3000.0, "ctrlrange": (-2.79,  2.79)},
    {"joint": "nova5joint4", "kp": 250.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint5", "kp": 200.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint6", "kp": 150.0, "ctrlrange": (-6.28,  6.28)},
]

# --- Expanded to 20 DoF per hand (same as your Genesis script) ---
RIGHT_HAND_PARAMS = [
    {"joint": "R_thumb_cmc_roll",   "kp": 40, "ctrlrange": (0, 1.0427)},
    {"joint": "R_thumb_cmc_yaw",    "kp": 40, "ctrlrange": (0, 1.2043)},
    {"joint": "R_thumb_cmc_pitch",  "kp": 35, "ctrlrange": (0, 0.5146)},
    {"joint": "R_thumb_mcp",        "kp": 30, "ctrlrange": (0, 0.7152)},
    {"joint": "R_thumb_ip",         "kp": 25, "ctrlrange": (0, 0.7763)},
    {"joint": "R_index_mcp_roll",   "kp": 30, "ctrlrange": (0, 0.2181)},
    {"joint": "R_index_mcp_pitch",  "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "R_index_pip",        "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "R_index_dip",        "kp": 25, "ctrlrange": (0, 1.8317)},
    {"joint": "R_middle_mcp_pitch", "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "R_middle_pip",       "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "R_middle_dip",       "kp": 25, "ctrlrange": (0, 0.6280)},
    {"joint": "R_ring_mcp_roll",    "kp": 30, "ctrlrange": (0, 0.2181)},
    {"joint": "R_ring_mcp_pitch",   "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "R_ring_pip",         "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "R_ring_dip",         "kp": 25, "ctrlrange": (0, 0.6280)},
    {"joint": "R_pinky_mcp_roll",   "kp": 25, "ctrlrange": (0, 0.3489)},
    {"joint": "R_pinky_mcp_pitch",  "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "R_pinky_pip",        "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "R_pinky_dip",        "kp": 25, "ctrlrange": (0, 0.6280)},
]
LEFT_HAND_PARAMS = [
    {"joint": "L_thumb_cmc_roll",   "kp": 40, "ctrlrange": (0, 1.0427)},
    {"joint": "L_thumb_cmc_yaw",    "kp": 40, "ctrlrange": (0, 1.2043)},
    {"joint": "L_thumb_cmc_pitch",  "kp": 35, "ctrlrange": (0, 0.5149)},
    {"joint": "L_thumb_mcp",        "kp": 30, "ctrlrange": (0, 0.7152)},
    {"joint": "L_thumb_ip",         "kp": 25, "ctrlrange": (0, 0.7763)},
    {"joint": "L_index_mcp_roll",   "kp": 30, "ctrlrange": (0, 0.2181)},
    {"joint": "L_index_mcp_pitch",  "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "L_index_pip",        "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "L_index_dip",        "kp": 25, "ctrlrange": (0, 0.6280)},
    {"joint": "L_middle_mcp_pitch", "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "L_middle_pip",       "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "L_middle_dip",       "kp": 25, "ctrlrange": (0, 1.6280)},
    {"joint": "L_ring_mcp_roll",    "kp": 30, "ctrlrange": (0, 0.2181)},
    {"joint": "L_ring_mcp_pitch",   "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "L_ring_pip",         "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "L_ring_dip",         "kp": 25, "ctrlrange": (0, 0.6280)},
    {"joint": "L_pinky_mcp_roll",   "kp": 25, "ctrlrange": (0, 0.3489)},
    {"joint": "L_pinky_mcp_pitch",  "kp": 35, "ctrlrange": (0, 1.3607)},
    {"joint": "L_pinky_pip",        "kp": 30, "ctrlrange": (0, 1.8317)},
    {"joint": "L_pinky_dip",        "kp": 25, "ctrlrange": (0, 0.6280)},
]

# ------------------------------------------------------------
# Tunables
# ------------------------------------------------------------
ARM_INPUT_IN_DEGREES = True
SPEEDUP = 2.0
DT_SIMUL = 0.01        # will set model.opt.timestep
DT_CMD   = 0.01        # command rate (playback cadence)

GROUP_DELAYS = {"n2": 0.0, "n5": 0.0, "lh": 0.0, "rh": 0.0}
REQUIRED_FINISH_GROUPS = ["n2", "n5"]     # or "ALL"
TAIL_SECONDS = 0.0

# Paths (edit to yours)
MJCF_PATH  = "./robot_urdf_genesis/scene_combo.xml"     # recommend: include draw.xml in this file via <include/>
OBJECT_MJCF_PATH = "./robot_urdf_genesis/draw.xml"  # optional; see note above
nova2_path = "exp_1/draw/draw_t1/nova2.txt"
nova5_path = "exp_1/draw/draw_t1/nova5.txt"
left_path  = "exp_1/draw/draw_t1/left.txt"
right_path = "exp_1/draw/draw_t1/right.txt"
video_out  = Path("renders") / "draw_t1_sim_mjwarp.mp4"

SHOW_VIEWER   = True    # sync CPU viewer with GPU state (optional)
RECORD_VIDEO  = True    # offscreen render via mujoco.Renderer (requires imageio + ffmpeg)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def params_to_arrays(group):
    names = [p["joint"] for p in group]
    kp    = np.array([p["kp"] for p in group], dtype=np.float32)
    lo    = np.array([p["ctrlrange"][0] for p in group], dtype=np.float64)
    hi    = np.array([p["ctrlrange"][1] for p in group], dtype=np.float64)
    return names, kp, lo, hi

def load_ros_txt_positions(txt_path, key="position"):
    t_list, q_list = [], []
    secs, nsecs = None, None
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("secs:"):
                try: secs = int(s.split(":", 1)[1].strip())
                except: secs = None
            elif s.startswith("nsecs:"):
                try: nsecs = int(s.split(":", 1)[1].strip())
                except: nsecs = 0
            elif s.startswith(f"{key}:"):
                arr = ast.literal_eval(s.split(":", 1)[1].strip())
                q_list.append(np.asarray(arr, dtype=np.float64))
                t_list.append((float(secs) + float(nsecs or 0)*1e-9) if secs is not None
                              else (t_list[-1] + 1.0 if t_list else 0.0))
    if not q_list:
        raise RuntimeError(f"No '{key}:' entries found in {txt_path}")
    t = np.asarray(t_list, dtype=np.float64)
    q = np.vstack(q_list)
    t -= t[0]
    uniq, idx = np.unique(t, return_index=True)
    return uniq, q[idx]

def maybe_to_rad(q, assume_degrees=True):
    return np.deg2rad(q).astype(q.dtype, copy=False) if assume_degrees else q

def resample_to_grid(t_src, q_src, t_eval, hold_ends=True):
    Ngrid, D = len(t_eval), q_src.shape[1]
    q = np.zeros((Ngrid, D), dtype=np.float64)
    left_vals  = q_src[0]  if hold_ends else np.full(D, np.nan)
    right_vals = q_src[-1] if hold_ends else np.full(D, np.nan)
    for j in range(D):
        q[:, j] = np.interp(t_eval, t_src, q_src[:, j], left=left_vals[j], right=right_vals[j])
    return q

def select_bounds(names_all, ok_names, lo_all, hi_all):
    name2i = {n: i for i, n in enumerate(names_all)}
    sel = [name2i[n] for n in ok_names]
    return lo_all[sel], hi_all[sel]

def process_hand_data(q_raw, hand_params, hand_prefix):
    """Map 10×(0..255) signals → 20 DoF angles in radians (inverse linear mapping)."""
    num_samples = q_raw.shape[0]
    num_joints  = len(hand_params)
    q_processed = np.zeros((num_samples, num_joints), dtype=np.float64)

    name_to_idx   = {p['joint']: i for i, p in enumerate(hand_params)}
    name_to_range = {p['joint']: p['ctrlrange'] for p in hand_params}

    input_map = {
        'thumb_flex': 0, 'thumb_yaw': 1, 'index_flex': 2, 'middle_flex': 3,
        'ring_flex': 4, 'pinky_flex': 5, 'index_roll': 6, 'ring_roll': 7,
        'pinky_roll': 8, 'thumb_roll': 9,
    }
    joint_mapping = {
        'thumb_flex':  [f'{hand_prefix}_thumb_cmc_pitch', f'{hand_prefix}_thumb_mcp', f'{hand_prefix}_thumb_ip'],
        'thumb_roll':  [f'{hand_prefix}_thumb_cmc_roll'],
        'thumb_yaw':   [f'{hand_prefix}_thumb_cmc_yaw'],
        'index_flex':  [f'{hand_prefix}_index_mcp_pitch', f'{hand_prefix}_index_pip', f'{hand_prefix}_index_dip'],
        'middle_flex': [f'{hand_prefix}_middle_mcp_pitch', f'{hand_prefix}_middle_pip', f'{hand_prefix}_middle_dip'],
        'ring_flex':   [f'{hand_prefix}_ring_mcp_pitch', f'{hand_prefix}_ring_pip', f'{hand_prefix}_ring_dip'],
        'pinky_flex':  [f'{hand_prefix}_pinky_mcp_pitch', f'{hand_prefix}_pinky_pip', f'{hand_prefix}_pinky_dip'],
        'index_roll':  [f'{hand_prefix}_index_mcp_roll'],
        'ring_roll':   [f'{hand_prefix}_ring_mcp_roll'],
        'pinky_roll':  [f'{hand_prefix}_pinky_mcp_roll'],
    }
    for input_name, joint_names in joint_mapping.items():
        s = q_raw[:, input_map[input_name]].astype(np.float64)
        for jn in joint_names:
            if jn in name_to_idx:
                jidx = name_to_idx[jn]
                lo, hi = name_to_range[jn]
                q_processed[:, jidx] = lo + ((255.0 - s) / 255.0) * (hi - lo)
    return q_processed

# -------- MuJoCo joint/DoF resolving (hinge/slide expected) --------
def resolve_dofs_mujoco(m: mujoco.MjModel, joint_names):
    """Return (dof_idx[], qpos_idx[], ok_names[], missing_names[])."""
    dof_idx, qpos_idx, ok, missing = [], [], [], []
    for name in joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            missing.append(name); continue
        jtype = m.jnt_type[jid]
        # handle hinge(=mjJNT_HINGE) / slide(=mjJNT_SLIDE) only
        if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            missing.append(name); continue
        dofadr  = int(m.jnt_dofadr[jid])
        qposadr = int(m.jnt_qposadr[jid])
        dof_idx.append(dofadr)
        qpos_idx.append(qposadr)
        ok.append(name)
    return np.asarray(dof_idx, dtype=np.int32), np.asarray(qpos_idx, dtype=np.int32), ok, missing

# -------- Warp kernel: PD torque for each targeted DoF --------
@wp.kernel
def pd_torque(
    # MJWarp device arrays are 2-D: (nworld, nq/nv)
    qpos: wp.array(dtype=float, ndim=2),
    qvel: wp.array(dtype=float, ndim=2),
    qpos_idx: wp.array(dtype=wp.int32),   # indices are 1-D
    dof_idx: wp.array(dtype=wp.int32),
    target: wp.array(dtype=float),        # 1-D (K,)
    kp: wp.array(dtype=float),            # 1-D (K,)
    kv: wp.array(dtype=float),            # 1-D (K,)
    tau: wp.array(dtype=float, ndim=2),   # (nworld, nv)
):
    i = wp.tid()
    world = 0  # using nworld=1; extend if you batch worlds

    q_i  = qpos[world, qpos_idx[i]]
    qd_i = qvel[world, dof_idx[i]]

    err = target[i] - q_i
    tau[world, dof_idx[i]] = kp[i] * err - kv[i] * qd_i



# ---------- kernels ----------
@wp.kernel
def zero2d(a: wp.array(dtype=float, ndim=2)):
    i, j = wp.tid()
    a[i, j] = 0.0

@wp.kernel
def scatter_ctrl(ctrl: wp.array(dtype=float, ndim=2),
                 act_idx: wp.array(dtype=int),
                 vals: wp.array(dtype=float)):
    # write ctrl values for a subset of actuators (nworld==1)
    i = wp.tid()
    ctrl[0, act_idx[i]] = vals[i]

# ---------- helper: joint id from qpos address (hinge/slide only) ----------
def joint_for_qpos(m: mujoco.MjModel, qpos_i: int) -> int:
    for jid in range(m.njnt):
        if m.jnt_qposadr[jid] == qpos_i:
            return jid
    return -1


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    wp.init()  # initialize Warp runtime

    # --- Load model (CPU) and create GPU copies ---
    mjm = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mjm.opt.timestep = DT_SIMUL  # match your Genesis dt
    d_cpu = mujoco.MjData(mjm)

    # Move to GPU (nworld=1). MJWarp mirrors the MuJoCo C API; step with mjw.step().
    # See docs for put_model/make_data/step and CUDA Graph capture recommendations. :contentReference[oaicite:2]{index=2}
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=1)

    # Optional: enable parallel line-search (can help convergence for harder contacts). :contentReference[oaicite:3]{index=3}
    try:
        m.opt.ls_parallel = True
    except Exception:
        pass

    # --- Resolve DoFs and per-group gains/limits ---
    groups = {"n2": NOVA2_PARAMS, "n5": NOVA5_PARAMS, "lh": LEFT_HAND_PARAMS, "rh": RIGHT_HAND_PARAMS}
    indices, qpos_indices, bounds, gains = {}, {}, {}, {}
    for key, params in groups.items():
        names, kp, lo, hi = params_to_arrays(params)
        dof_idx, qpos_idx, ok, missing = resolve_dofs_mujoco(mjm, names)
        if missing:
            print(f"[Warn] Missing or unsupported joints [{key}]: {missing}")
        if dof_idx.size == 0:
            print(f"[Skip] {key}: no resolved hinge/slide DoFs.")
            continue
        lo_sel, hi_sel = select_bounds(names, ok, lo, hi)
        indices[key]      = dof_idx
        qpos_indices[key] = qpos_idx
        bounds[key]       = (lo_sel.astype(np.float64), hi_sel.astype(np.float64))
        gains[key]        = kp[:len(ok)].astype(np.float32)

    # --- Load trajectories and preprocess (arms: deg->rad; hands: 0-255->angles) ---
    data_paths = {"n2": nova2_path, "n5": nova5_path, "lh": left_path, "rh": right_path}
    loaded = {}
    for key, path in data_paths.items():
        if key not in indices:
            continue
        p = Path(path)
        if not p.exists():
            print(f"[Skip] {key}: file not found -> {path}")
            continue
        t_src, q_src = load_ros_txt_positions(path)
        if key == "lh":
            q_src = process_hand_data(q_src, LEFT_HAND_PARAMS, 'L')
        elif key == "rh":
            q_src = process_hand_data(q_src, RIGHT_HAND_PARAMS, 'R')
        else:
            q_src = maybe_to_rad(q_src, assume_degrees=ARM_INPUT_IN_DEGREES)
        loaded[key] = (t_src, q_src)
        print(f"[Load] {key}: {len(t_src)} samples, D={q_src.shape[1]}, duration={t_src[-1]:.3f}s from {path}")

    if not loaded:
        raise RuntimeError("No playable groups found (missing files or no DoFs resolved).")

    # --- Build unified playback grid ensuring required groups finish ---
    S = max(SPEEDUP, 1.0)

    def need_seconds(k, t_src):
        return float(t_src[-1])/S + float(GROUP_DELAYS.get(k, 0.0))

    if REQUIRED_FINISH_GROUPS == "ALL":
        keys_for_stop = list(loaded.keys())
    else:
        ksel = [k for k in (REQUIRED_FINISH_GROUPS or []) if k in loaded]
        keys_for_stop = ksel if ksel else list(loaded.keys())

    T_playback = max(need_seconds(k, loaded[k][0]) for k in keys_for_stop)
    steps_per_cmd = max(1, int(round(DT_CMD / DT_SIMUL)))
    t_grid_play = np.arange(0.0, T_playback + 1e-9, DT_CMD, dtype=np.float64)
    t_eval_src_base = t_grid_play * S
    TARGET_STEPS = len(t_grid_play)
    print(f"[Sync] Finish-for={keys_for_stop} | SPEEDUP={SPEEDUP:.2f}x | steps={TARGET_STEPS} | play≈{TARGET_STEPS*DT_CMD:.2f}s")

    # --- Prepare resampled commands, apply per-group delay, align DoFs, clamp ---
    q_cmds = {}              # key -> (q_cmd[T, D_k], dof_idx[K], qpos_idx[K], kp[K], kv[K])
    for key, (t_src, q_src) in loaded.items():
        delay = GROUP_DELAYS.get(key, 0.0)
        t_eval_src = t_eval_src_base - delay*S
        q = resample_to_grid(t_src, q_src, t_eval_src, hold_ends=True)

        dof_idx  = indices[key]
        qpos_idx = qpos_indices[key]
        D_target = dof_idx.size
        if q.shape[1] != D_target:
            q_aligned = np.zeros((TARGET_STEPS, D_target), dtype=q.dtype)
            mcol = min(q.shape[1], D_target)
            q_aligned[:, :mcol] = q[:, :mcol]
            q = q_aligned
            print(f"[Dim] {key}: mapped {q.shape[1]} -> {D_target} columns")

        lo, hi = bounds[key]
        q = np.minimum(np.maximum(q, lo[None, :]), hi[None, :])
        kp = gains[key].astype(np.float32)
        kv = (2.0 * np.sqrt(np.maximum(kp, 1e-6))).astype(np.float32)
        q_cmds[key] = (q, dof_idx.astype(np.int32), qpos_idx.astype(np.int32), kp, kv)

    if not q_cmds:
        raise RuntimeError("Nothing to play after alignment.")

    # --- Flatten all groups into one PD target vector order ---
    order = list(q_cmds.keys())
    all_dof_idx  = np.concatenate([q_cmds[k][1] for k in order], axis=0)
    all_qpos_idx = np.concatenate([q_cmds[k][2] for k in order], axis=0)
    all_kp       = np.concatenate([q_cmds[k][3] for k in order], axis=0)
    all_kv       = np.concatenate([q_cmds[k][4] for k in order], axis=0)

    # Pre-allocate device arrays
    device = "cuda"  # change to "cpu" to debug on CPU
    d_target = wp.zeros(len(all_dof_idx), dtype=float, device=device)
    d_kp     = wp.from_numpy(all_kp.astype(np.float32), dtype=float, device=device)
    d_kv     = wp.from_numpy(all_kv.astype(np.float32), dtype=float, device=device)
    d_dof    = wp.from_numpy(all_dof_idx.astype(np.int32), dtype=int, device=device)
    d_qposi  = wp.from_numpy(all_qpos_idx.astype(np.int32), dtype=int, device=device)

    # --- Move to initial pose and settle (GPU) ---
    # Set qpos on GPU from the 1st sample of each group (simple host->device copy).
    # MJWarp arrays are on device; we copy via wp.copy/array; call mjw.step() to settle. :contentReference[oaicite:4]{index=4}
    # Build a host qpos (size nq) with current defaults, then overwrite targeted qpos indices
    qpos0 = d.qpos.numpy()               # shape (1, nq)
    qpos0 = qpos0.copy()
    start_targets = np.concatenate([q_cmds[k][0][0] for k in order], axis=0)
    qpos0[0, all_qpos_idx] = start_targets
    # write back
    wp.copy(d.qpos, wp.from_numpy(qpos0, dtype=float, device=device))
    # zero velocities
    d.qvel.fill_(0.0)  # Warp arrays support fill_ helper. :contentReference[oaicite:5]{index=5}
    # settle a bit
    for _ in range(30):
        mjw.step(m, d)

    # --- Optional viewer & recorder (CPU-side) ---
    viewer = None
    renderer = None
    writer = None
    if SHOW_VIEWER:
        # Passive viewer displays CPU mjData; we mirror GPU state each frame via qpos/qvel and mj_forward. :contentReference[oaicite:6]{index=6}
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(mjm, d_cpu)
    # if RECORD_VIDEO and imageio is not None:
    #     video_out.parent.mkdir(parents=True, exist_ok=True)
    #     renderer = mujoco.Renderer(mjm, 960, 640)      # simple offscreen renderer (see Python bindings doc). :contentReference[oaicite:7]{index=7}
    #     writer = imageio.get_writer(str(video_out), fps=int(round(1.0/DT_SIMUL)))

    # --- Capture kernels + step in a CUDA graph (faster repeated launches). :contentReference[oaicite:8]{index=8}
    act_idx_by_group = {}
    for key in order:
        q, dof_idx, qpos_idx, *_ = q_cmds[key]         # we only need q & qpos_idx
        act_idx = np.full(qpos_idx.shape[0], -1, dtype=np.int32)
        for j, qp in enumerate(qpos_idx):
            jid = joint_for_qpos(mjm, int(qp))
            if jid >= 0:
                # find an actuator that targets this joint
                for a in range(mjm.nu):
                    # JOINT transform; actuator_trnid[a,0] stores the joint id
                    if (mjm.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and
                        mjm.actuator_trnid[a, 0] == jid):
                        act_idx[j] = a
                        break
        miss = np.where(act_idx < 0)[0]
        if len(miss):
            print(f"[Warn] {key}: {len(miss)} DoFs have no actuator; skipped.")
        act_idx_by_group[key] = act_idx

    # Build a flattened selector so we can gather only actuated columns each tick
    flat_keep_idx, act_keep_idx = [], []
    offset = 0
    for key in order:
        q, *_ = q_cmds[key]
        act_idx = act_idx_by_group[key]
        for j, a in enumerate(act_idx):
            if a >= 0:
                flat_keep_idx.append(offset + j)   # column in the flattened q vector
                act_keep_idx.append(int(a))        # actuator index to write
        offset += q.shape[1]

    flat_keep_idx = np.asarray(flat_keep_idx, dtype=np.int32)
    act_keep_idx  = np.asarray(act_keep_idx,  dtype=np.int32)
    K = len(act_keep_idx)
    print(f"[ActMap] driving {K} actuators out of {offset} DoFs (nu={mjm.nu})")

    # Precompute ctrl ranges for the actuators we actually drive
    ctrl_lo_keep = mjm.actuator_ctrlrange[act_keep_idx, 0]
    ctrl_hi_keep = mjm.actuator_ctrlrange[act_keep_idx, 1]

    # Device buffers for fast per-step updates
    d_act_keep_idx = wp.from_numpy(act_keep_idx, dtype=int, device=device)
    d_ctrl_vals    = wp.zeros(K, dtype=float, device=device)    # updated each tick

    # ---------- playback loop using d.ctrl ----------
    print("\n[Run] Starting d.ctrl playback (rendering every substep)...")
    for i in range(TARGET_STEPS):
        # Flatten in the same concatenation order you use elsewhere
        q_flat = np.concatenate([q_cmds[k][0][i] for k in order], axis=0)

        # Pick only actuated entries and clamp to ctrl range
        vals = q_flat[flat_keep_idx].astype(np.float32, copy=False)
        if np.isfinite(ctrl_lo_keep).all() and np.isfinite(ctrl_hi_keep).all():
            np.clip(vals, ctrl_lo_keep, ctrl_hi_keep, out=vals)

        # Upload control values for this tick
        wp.copy(d_ctrl_vals, wp.from_numpy(vals, dtype=float, device=device))

        # Zero all controls, then scatter our subset (GPU)
        wp.launch(zero2d, dim=(1, mjm.nu), inputs=[d.ctrl], device=device)
        wp.launch(scatter_ctrl, dim=K, inputs=[d.ctrl, d_act_keep_idx, d_ctrl_vals], device=device)

        # Physics substeps + stepwise rendering
        for _ in range(steps_per_cmd):
            mjw.step(m, d)

            if viewer or writer:
                # mirror GPU -> CPU for viewer/offscreen render
                d_cpu.qpos[:] = d.qpos.numpy()[0]
                d_cpu.qvel[:] = d.qvel.numpy()[0]
                mujoco.mj_forward(mjm, d_cpu)
                if viewer:
                    viewer.sync()
                if writer and renderer is not None:
                    frame = renderer.render(mjm, d_cpu)  # HxWx3 uint8
                    writer.append_data(frame)

    # Optional tail settle
    for _ in range(int(TAIL_SECONDS / DT_SIMUL)):
        wp.launch(zero2d, dim=(1, mjm.nu), inputs=[d.ctrl], device=device)
        mjw.step(m, d)
        if viewer or writer:
            d_cpu.qpos[:] = d.qpos.numpy()[0]; d_cpu.qvel[:] = d.qvel.numpy()[0]
            mujoco.mj_forward(mjm, d_cpu)
            if viewer: viewer.sync()
            if writer and renderer is not None:
                frame = renderer.render(mjm, d_cpu)
                writer.append_data(frame)

    if writer:
        writer.close()
        print(f"[OK] Saved video to: {video_out.resolve()}")
    if viewer:
        viewer.close()