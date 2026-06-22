#!/usr/bin/env python3
"""
isaacgym_play_motion.py

在 IsaacGym 无重力模式下播放动作轨迹（以 Walking 为例）。

支持两种输入格式：
  1. qpos.npy  — 直接播放关节位置（qpos 格式: pos+rot+joints）
  2. MotionLoaderNingTracking JSON — 使用环境内置的 motion_loader

无重力模式：gym.set_actor_root_state_tensor + 不施加驱动力，让机器人靠初始状态保持姿态。

用法:
  # 方式1: 从 qpos.npy 播放（直接设置关节位置）
  python isaacgym_play_motion.py --source qpos --motion walking/Walking_3

  # 方式2: 从 JSON 播放（通过环境内置 motion_loader）
  python isaacgym_play_motion.py --source json --motion walking/Walking_3

  # 指定环境数量和速度
  python isaacgym_play_motion.py --source qpos --motion walking/Walking_3 --num_envs 1 --speed 1.0
"""

import argparse
import os
import sys
import time
from pathlib import Path

# IsaacGym MUST be imported before torch
sys.path.insert(0, str(Path(__file__).parent))
import isaacgym
from isaacgym import gymapi, gymtorch

import numpy as np
import torch

from humanoid.envs import *
from humanoid.utils import get_args, task_registry


# =============================================================================
# Helpers
# =============================================================================

def quat_mul(q1, q2):
    """Multiply two quaternions (x,y,z,w)."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([x, y, z, w], axis=-1)


def normalize_quat(q):
    """Normalize quaternion."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + 1e-8)


def euler_to_quat_xyzw(euler):
    """Euler (r, p, y) -> quaternion (x, y, z, w). ZYX order."""
    r, p, y = euler[..., 0], euler[..., 1], euler[..., 2]
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.stack([x, y, z, w], axis=-1)


def load_qpos_motion(motion_path, root_dir):
    """Load motion from qpos.npy. Auto-detects quat (25 cols) vs euler (24 cols)."""
    motion_dir = Path(root_dir) / Path(motion_path).parent
    motion_name = Path(motion_path).name
    qpos_path = motion_dir / (motion_name + '_qpos.npy')
    mujoco_path = motion_dir / (motion_name + '_mujoco.json')

    import json
    qpos = np.load(str(qpos_path))
    with open(str(mujoco_path)) as f:
        mj_data = json.load(f)
    fps = float(mj_data['fps'])

    n_frames, n_cols = qpos.shape
    if n_cols == 25:
        root_pos = qpos[:, 0:3]
        root_rot = qpos[:, 3:7]    # xyzw
        dof_pos = qpos[:, 7:25]
    elif n_cols == 24:
        root_pos = qpos[:, 0:3]
        euler = qpos[:, 3:6]
        dof_pos = qpos[:, 6:24]
        root_rot = euler_to_quat_xyzw(euler)
    else:
        raise ValueError(f"Unsupported qpos: {n_cols} columns")

    print(f"  Loaded: {n_frames} frames, fps={fps:.2f}, format={'quat' if n_cols==25 else 'euler'}")
    print(f"  root_pos: [{root_pos.min():.3f}, {root_pos.max():.3f}]")
    print(f"  dof_pos range: [{dof_pos.min():.3f}, {dof_pos.max():.3f}]")

    return {
        'root_pos': root_pos,
        'root_rot': root_rot,
        'dof_pos': dof_pos,
        'fps': fps,
        'n_frames': n_frames,
        'dt': 1.0 / fps,
    }


# =============================================================================
# Main playback
# =============================================================================

def play_qpos(args):
    """Play motion directly from qpos.npy, bypassing RL.

    Direct position control: each step directly set dof_pos/dof_vel and
    root_states in the physics engine. Bypasses PD torque controller.
    """
    import json

    print(f"\n[{'='*50}]")
    print(f"[Play] Loading motion from qpos.npy")
    print(f"[Play] Source: {args.motion}")

    motion = load_qpos_motion(args.motion, args.root)

    print(f"\n[Play] Creating IsaacGym environment...")
    args.num_envs = min(args.num_envs, motion['n_frames'])
    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = 9999
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.sim.gravity = [0.0, 0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    sim_params = env.gym.get_sim_params(env.sim)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    env.gym.set_sim_params(env.sim, sim_params)

    num_dof = env.num_dof
    print(f"[Play] DOF: {num_dof}, num_envs: {args.num_envs}")

    env_frame_idx = np.arange(args.num_envs) % motion['n_frames']

    dt = motion['dt']
    root_pos = motion['root_pos']
    root_lin_vel = np.zeros_like(root_pos)
    root_lin_vel[1:-1] = (root_pos[2:] - root_pos[:-2]) / (2 * dt)
    root_lin_vel[0] = (root_pos[1] - root_pos[0]) / dt
    root_lin_vel[-1] = (root_pos[-1] - root_pos[-2]) / dt

    dof_vel = np.zeros_like(motion['dof_pos'])
    dof_vel[1:-1] = (motion['dof_pos'][2:] - motion['dof_pos'][:-2]) / (2 * dt)
    dof_vel[0] = (motion['dof_pos'][1] - motion['dof_pos'][0]) / dt
    dof_vel[-1] = (motion['dof_pos'][-1] - motion['dof_pos'][-2]) / dt

    dof_state = env.dof_state

    print(f"\n[Play] Initial state:")
    print(f"  root_pos[0]: {root_pos[0]}")
    print(f"  dof_pos[0][:5]: {motion['dof_pos'][0][:5]}")
    print(f"\n[Play] Starting playback at {args.speed}x speed...")
    print(f"[Play] Method: direct position control (no PD torque)")

    try:
        frame_time = motion['dt'] / args.speed
        i_frame = 0

        while True:
            loop_start = time.time()

            idx = env_frame_idx

            for e in range(args.num_envs):
                fi = idx[e]
                dof_state[e, 0::2] = torch.from_numpy(motion['dof_pos'][fi]).to(env.device)
                dof_state[e, 1::2] = torch.from_numpy(dof_vel[fi] * args.speed).to(env.device)

            env.root_states[:, 0:3] = torch.from_numpy(root_pos[idx]).to(env.device)
            env.root_states[:, 3:7] = torch.from_numpy(motion['root_rot'][idx]).to(env.device)
            env.root_states[:, 7:10] = torch.from_numpy(root_lin_vel[idx] * args.speed).to(env.device)
            env.root_states[:, 10:13] = 0.0

            root_state_t = gymtorch.unwrap_tensor(env.root_states)
            env.gym.set_actor_root_state_tensor(env.sim, root_state_t)

            dof_state_t = gymtorch.unwrap_tensor(dof_state)
            env.gym.set_dof_state_tensor(env.sim, dof_state_t)

            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)

            env_frame_idx = (env_frame_idx + 1) % motion['n_frames']
            i_frame += 1

            if i_frame % 100 == 0:
                root = env.root_states[0].cpu().numpy()
                print(f"  Step {i_frame}/{motion['n_frames']}: "
                      f"root_z={root[2]:+.4f}, root_vel_z={root[9]:+.4f}")

            if env.viewer is not None:
                env.gym.step_graphics(env.sim)
                env.gym.draw_viewer(env.viewer, env.sim, False)
                env.gym.sync_frame_time(env.sim)

            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Play] Interrupted.")
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        print("[Play] Done.")


def play_json(args):
    """Play motion through the environment's built-in motion_loader.

    Uses direct position control: each step we read reference state from
    motion_loader and directly set dof_pos / root_states in the physics
    engine. This bypasses the PD torque controller, so zero-gravity
    has no effect on the robot.
    """
    print(f"\n[{'='*50}]")
    print(f"[Play] Playing via environment motion_loader")
    print(f"[Play] Motion: {args.motion}")

    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    # Set motion file
    motion_path = os.path.join(args.root, args.motion + '.json')
    env_cfg.motion_loader.reference_motion_file = [motion_path]
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 9999
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.env.test = True
    env_cfg.sim.gravity = [0.0, 0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    sim_params = env.gym.get_sim_params(env.sim)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    env.gym.set_sim_params(env.sim, sim_params)

    # Reset (sets tensors but physics engine needs simulate() to propagate)
    obs = env.get_observations()

    # Force physics engine to propagate initial state from reset
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)

    print(f"[Play] Initial state after reset:")
    root = env.root_states[0].cpu().numpy()
    dof = env.dof_pos[0].cpu().numpy()
    print(f"  root_pos: [{root[0]:+.4f}, {root[1]:+.4f}, {root[2]:+.6f}]")
    print(f"  root_linvel: [{root[7]:+.4f}, {root[8]:+.4f}, {root[9]:+.4f}]")
    print(f"  dof_pos[:6]: {dof[:6]}")
    print(f"  default_dof_pos[:6]: {env.default_dof_pos[0].cpu().numpy()[:6]}")
    print(f"  ref_dof_pos[:6]: {env.ref_dof_pos[0].cpu().numpy()[:6]}")

    print(f"\n[Play] Starting playback at {args.speed}x speed...")
    print(f"[Play] Method: direct position control (no PD torque)")

    try:
        frame_dt = env.dt / args.speed
        last_render = time.time()
        step_count = 0

        while True:
            loop_start = time.time()

            # Compute reference state from motion_loader
            env.compute_ref_state()
            phase = env._get_phase()
            motion_times = (phase * env.motion_lenth).clamp(
                min=0.0, max=env.motion_lenth - env.dt
            ).cpu().numpy()

            traj_idxs = np.array([0] * env.num_envs)
            frames = env.motion_loader.get_full_frame_at_time_batch(
                traj_idxs, motion_times
            )

            # Root state
            ref_root_pos = env.motion_loader_class.get_root_pos_batch(frames)
            ref_root_rot = env.motion_loader_class.get_root_rot_batch(frames)
            ref_lin_vel = env.motion_loader_class.get_linear_vel_batch(frames)
            ref_ang_vel = env.motion_loader_class.get_angular_vel_batch(frames)
            ref_dof_pos = env.motion_loader_class.get_joint_pose_batch(frames)
            ref_dof_vel = env.motion_loader_class.get_joint_vel_batch(frames)

            # Directly set root state (pos in world frame)
            env.root_states[:, 0:3] = ref_root_pos
            env.root_states[:, 3:7] = ref_root_rot
            env.root_states[:, 7:10] = ref_lin_vel * args.speed
            env.root_states[:, 10:13] = ref_ang_vel * args.speed

            # Directly set DOF positions
            env.dof_pos[:] = ref_dof_pos
            env.dof_vel[:] = ref_dof_vel * args.speed

            # Push to physics engine
            root_state_t = gymtorch.unwrap_tensor(env.root_states)
            env.gym.set_actor_root_state_tensor(env.sim, root_state_t)

            dof_state_t = gymtorch.unwrap_tensor(env.dof_state)
            env.gym.set_dof_state_tensor(env.sim, dof_state_t)

            # Physics step
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)

            step_count += 1
            if step_count % 100 == 0:
                root = env.root_states[0].cpu().numpy()
                print(f"  Step {step_count}: root_z={root[2]:+.4f}, "
                      f"root_vel_z={root[9]:+.4f}")

            # Render
            if env.viewer is not None:
                now = time.time()
                if now - last_render > 1 / 30:
                    env.gym.step_graphics(env.sim)
                    env.gym.draw_viewer(env.viewer, env.sim, False)
                    env.gym.sync_frame_time(env.sim)
                    last_render = now

            elapsed = time.time() - loop_start
            sleep_time = frame_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Play] Interrupted.")
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        print("[Play] Done.")


if __name__ == '__main__':
    # Parse playback-specific args first (before get_args consumes them)
    import argparse
    motion_parser = argparse.ArgumentParser(add_help=False)
    motion_parser.add_argument('--source', type=str, default='qpos',
                              choices=['qpos', 'json'])
    motion_parser.add_argument('--motion', type=str, default='walking/Walking_3')
    motion_parser.add_argument('--root', type=str,
                              default='/home/fff/noetix/noetix_n2_gym/datasets/mocap_motions/ning')
    motion_parser.add_argument('--speed', type=float, default=1.0)
    motion_args, remaining = motion_parser.parse_known_args()

    # Build args for IsaacGym (with motion args removed)
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()

    # Override with task/environment settings
    args.task = 'n2_mimic'
    args.num_envs = 1
    args.headless = False
    args.motion = motion_args.motion
    args.root = motion_args.root
    args.speed = motion_args.speed

    if motion_args.source == 'qpos':
        play_qpos(args)
    else:
        play_json(args)
