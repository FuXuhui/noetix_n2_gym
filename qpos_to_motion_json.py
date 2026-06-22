#!/usr/bin/env python3
"""
Convert raw motion trajectories into MotionLoaderNingTracking JSON.

Supported inputs:
- qpos/mujoco pair: <stem>_qpos.npy + <stem>_mujoco.json
- frames json:      <stem>_frames.json

Output format consumed by MotionLoaderNingTracking:
  [0:3]    root_pos
  [3:7]    root_rot (xyzw)
  [7:25]   dof_pos (18)
  [25:85]  body_pos_local (20 bodies × 3, including world body at index 0)
  [85:88]  root_lin_vel
  [88:91]  root_ang_vel
  [91:109] joint_vel (18)
  [109:111] contact_mask (2 feet)

Notes:
- MuJoCo's `model.nbody` here is 20 and includes the `world` body at index 0.
- We intentionally keep the body section at 20 × 3 = 60 columns to match
  `MotionLoaderNingTracking.TAR_TOE_POS_LOCAL_SIZE = 60`.
- The world body local position is forced to zero to avoid leaking `-root_pos`
  into the reference body features.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

JOINT_NAMES = [
    "L_arm_shoulder_pitch_joint",
    "L_arm_shoulder_roll_joint",
    "L_arm_shoulder_yaw_joint",
    "L_arm_elbow_joint",
    "L_leg_hip_yaw_joint",
    "L_leg_hip_roll_joint",
    "L_leg_hip_pitch_joint",
    "L_leg_knee_joint",
    "L_leg_ankle_joint",
    "R_arm_shoulder_pitch_joint",
    "R_arm_shoulder_roll_joint",
    "R_arm_shoulder_yaw_joint",
    "R_arm_elbow_joint",
    "R_leg_hip_yaw_joint",
    "R_leg_hip_roll_joint",
    "R_leg_hip_pitch_joint",
    "R_leg_knee_joint",
    "R_leg_ankle_joint",
]

EXPECTED_BODY_COUNT = 20
EXPECTED_BODY_COLS = EXPECTED_BODY_COUNT * 3
EXPECTED_FRAME_LEN = 3 + 4 + 18 + EXPECTED_BODY_COLS + 3 + 3 + 18 + 2


def finite_diff(seq, dt):
    vel = np.zeros_like(seq)
    vel[1:-1] = (seq[2:] - seq[:-2]) / (2 * dt)
    vel[0] = (seq[1] - seq[0]) / dt
    vel[-1] = (seq[-1] - seq[-2]) / dt
    return vel


def euler_to_quat_xyzw(euler):
    r, p, y = euler
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])


def angular_vel_from_quat(quats_xyzw, dt):
    n = quats_xyzw.shape[0]
    ang_vel = np.zeros((n, 3))
    for i in range(n):
        q0 = quats_xyzw[0] if i == 0 else quats_xyzw[i - 1]
        q1 = quats_xyzw[-1] if i == n - 1 else quats_xyzw[i + 1]
        dq = (q1 - q0) / (2 * dt)
        qw, qx, qy, qz = q0[3], q0[0], q0[1], q0[2]
        dqx, dqy, dqz = dq[0], dq[1], dq[2]
        ang_vel[i] = [
            2 * (qw * dqx + qy * dqz - qz * dqy),
            2 * (qw * dqy + qz * dqx - qx * dqz),
            2 * (qw * dqz + qx * dqy - qy * dqx),
        ]
    return ang_vel


def build_mujoco_fk(xml_path):
    model = mj.MjModel.from_xml_path(xml_path)
    body_names = {}
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name:
            body_names[name] = i
    return model, body_names


def get_body_local_positions(model, qpos_mj):
    """Get all body positions in root-local coordinates.

    Fix A:
    - keep world body in the 20-body layout for loader compatibility
    - force world body local position to zero instead of (-root_pos)
    """
    data = mj.MjData(model)
    data.qpos[:] = qpos_mj
    mj.mj_fwdPosition(model, data)

    root_pos = qpos_mj[:3]
    body_pos_local = np.zeros((model.nbody, 3))
    for i in range(model.nbody):
        body_pos_local[i] = data.xpos[i] - root_pos

    if model.nbody > 0:
        body_pos_local[0] = 0.0

    return body_pos_local


def load_from_qpos(qpos_path, mujoco_path):
    qpos = np.load(qpos_path)
    n_frames, n_qpos = qpos.shape

    with open(mujoco_path) as f:
        mj_data = json.load(f)
    fps = float(mj_data["fps"])
    dt = 1.0 / fps

    if n_qpos == 25:
        root_pos = qpos[:, 0:3]
        root_rot = qpos[:, 3:7]
        dof_pos = qpos[:, 7:25]
    elif n_qpos == 24:
        root_pos = qpos[:, 0:3]
        euler = qpos[:, 3:6]
        dof_pos = qpos[:, 6:24]
        root_rot = np.array([euler_to_quat_xyzw(e) for e in euler])
    else:
        raise ValueError(f"Unsupported qpos shape (N, {n_qpos}). Expected 25 or 24.")

    return root_pos, root_rot, dof_pos, dt, n_frames


def load_from_frames(input_path, fps):
    with open(input_path) as f:
        frames_dict = json.load(f)

    n_frames = len(frames_dict)
    root_pos = np.zeros((n_frames, 3))
    root_rot = np.zeros((n_frames, 4))
    dof_pos = np.zeros((n_frames, 18))

    for i, frame in enumerate(frames_dict):
        root_pos[i] = frame["base_link"]["position"]
        root_rot[i] = frame["base_link"]["orientation"]
        for j, name in enumerate(JOINT_NAMES):
            dof_pos[i, j] = frame["joints"].get(name, 0.0)

    dt = 1.0 / fps
    return root_pos, root_rot, dof_pos, dt, n_frames


def convert(root_pos, root_rot, dof_pos, dt, xml_path, output_path, motion_name="Motion", motion_weight=0.5):
    n_frames = root_pos.shape[0]

    print(f"\n[{'=' * 50}]")
    print(f"[{motion_name}] frames={n_frames}, dt={dt:.6f}s, duration={n_frames * dt:.2f}s")
    print(f"[{motion_name}] Building MuJoCo model from {xml_path}...")

    model, body_names = build_mujoco_fk(xml_path)
    print(f"[{motion_name}] MuJoCo: nq={model.nq}, nbody={model.nbody}")
    if model.nbody != EXPECTED_BODY_COUNT:
        raise ValueError(
            f"Expected MuJoCo body count {EXPECTED_BODY_COUNT}, got {model.nbody}. "
            "Update converter and loader together before proceeding."
        )

    qpos_mj = np.zeros((n_frames, 25))
    qpos_mj[:, 0:3] = root_pos
    qpos_mj[:, 3:7] = root_rot
    qpos_mj[:, 7:25] = dof_pos

    print(f"[{motion_name}] Computing FK for {n_frames} frames...")
    all_body_local = np.zeros((n_frames, model.nbody, 3))
    for i in range(n_frames):
        all_body_local[i] = get_body_local_positions(model, qpos_mj[i])
        if (i + 1) % 100 == 0:
            print(f"[{motion_name}]   FK: {i + 1}/{n_frames}")

    print(f"[{motion_name}] Computing velocities...")
    root_lin_vel = finite_diff(root_pos, dt)
    ang_vel = angular_vel_from_quat(root_rot, dt)
    joint_vel = finite_diff(dof_pos, dt)

    # Fix B: keep body section at exactly 20 * 3 = 60 cols.
    # This preserves loader alignment, so C should no longer be needed.
    L_ankle_body_id = body_names.get("L_leg_ankle_link")
    R_ankle_body_id = body_names.get("R_leg_ankle_link")
    left_contact = np.zeros(n_frames)
    right_contact = np.zeros(n_frames)
    if L_ankle_body_id is not None:
        left_z = all_body_local[:, L_ankle_body_id, 2]
        left_contact = (left_z < 0.05).astype(float)
        print(f"[{motion_name}] L_contact frames: {int(left_contact.sum())}/{n_frames}")
    if R_ankle_body_id is not None:
        right_z = all_body_local[:, R_ankle_body_id, 2]
        right_contact = (right_z < 0.05).astype(float)
        print(f"[{motion_name}] R_contact frames: {int(right_contact.sum())}/{n_frames}")

    print(f"[{motion_name}] Building output frames...")
    frames = []
    for i in range(n_frames):
        body_local_flat = all_body_local[i].flatten()
        if len(body_local_flat) != EXPECTED_BODY_COLS:
            raise ValueError(
                f"Frame {i}: expected {EXPECTED_BODY_COLS} body columns, got {len(body_local_flat)}"
            )

        frame = np.concatenate([
            root_pos[i],
            root_rot[i],
            dof_pos[i],
            body_local_flat,
            root_lin_vel[i],
            ang_vel[i],
            joint_vel[i],
            np.array([left_contact[i], right_contact[i]]),
        ])
        if len(frame) != EXPECTED_FRAME_LEN:
            raise ValueError(f"Frame {i}: {len(frame)} != {EXPECTED_FRAME_LEN}")
        frames.append(frame.tolist())

    fp = np.array(frames[0])
    print(f"[{motion_name}] Frame length: {len(fp)}")
    print(f"[{motion_name}] frame0 body[0] (world body local): {fp[25:28].tolist()}")
    print(f"[{motion_name}] frame0 lin_vel: {fp[85:88].tolist()}")
    print(f"[{motion_name}] frame0 ang_vel: {fp[88:91].tolist()}")
    print(f"[{motion_name}] frame0 contact: {fp[109:111].tolist()}")

    motion_data = {
        "LoopMode": "Wrap",
        "FrameDuration": dt,
        "EnableCycleOffsetPosition": "true",
        "EnableCycleOffsetRotation": "true",
        "MotionWeight": motion_weight,
        "Frames": frames,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(motion_data, f)

    print(f"[{motion_name}] Saved {n_frames} frames to {output_path}")
    return output_path


def resolve_dataset_path(root, value, suffix=""):
    path = Path(value)
    if path.is_absolute():
        resolved = path
    else:
        resolved = Path(root) / path
    if suffix and not str(resolved).endswith(suffix):
        resolved = Path(str(resolved) + suffix)
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Convert motion data to MotionLoaderNingTracking JSON")
    parser.add_argument("--mode", choices=["qpos", "frames"], default="qpos")
    parser.add_argument("--motion", type=str, default="walking/Walking_3",
                        help="Motion stem under dataset root, e.g. walking/Walking_3")
    parser.add_argument("--input", type=str,
                        help="Optional explicit input path for frames mode")
    parser.add_argument("--output", type=str,
                        help="Optional explicit output path")
    parser.add_argument("--root", type=str,
                        default="/home/fff/noetix/noetix_n2_gym/datasets/mocap_motions/ning",
                        help="Dataset root directory")
    parser.add_argument("--xml", type=str,
                        default="/home/fff/GMR/assets/N2/mjcf/n2_18dof.xml",
                        help="MuJoCo XML file for FK")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frames mode only: motion FPS")
    parser.add_argument("--weight", type=float, default=0.5,
                        help="Motion weight")
    args = parser.parse_args()

    motion_stem = Path(args.motion).name
    motion_dir = Path(args.root) / Path(args.motion).parent

    if args.mode == "qpos":
        qpos_path = Path(str(motion_dir / motion_stem) + "_qpos.npy")
        mujoco_path = Path(str(motion_dir / motion_stem) + "_mujoco.json")
        output_path = Path(args.output) if args.output else Path(str(motion_dir / motion_stem) + ".json")

        for path, label in [(qpos_path, "qpos"), (mujoco_path, "mujoco.json")]:
            if not path.exists():
                print(f"ERROR: {label} not found at {path}")
                sys.exit(1)

        root_pos, root_rot, dof_pos, dt, _ = load_from_qpos(qpos_path, mujoco_path)
    else:
        input_path = resolve_dataset_path(args.root, args.input or f"{args.motion}_frames", ".json")
        output_path = Path(args.output) if args.output else resolve_dataset_path(args.root, args.motion, ".json")
        if not input_path.exists():
            print(f"ERROR: input not found at {input_path}")
            sys.exit(1)
        root_pos, root_rot, dof_pos, dt, _ = load_from_frames(input_path, args.fps)

    convert(
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        dt=dt,
        xml_path=args.xml,
        output_path=str(output_path),
        motion_name=motion_stem,
        motion_weight=args.weight,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
