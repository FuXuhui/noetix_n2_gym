"""
sim2sim_motion.py — sim2sim for N2 walking/boxing motion imitation policies.

Loads a walking3 (or similar GMR-tracking) ONNX policy exported from IsaacGym,
runs it in MuJoCo, and tracks the motion reference with keyboard velocity override.

Usage:
    python sim2sim_motion.py --config_file n2_motion.yaml

Keyboard controls (override motion velocity):
    Up/Down    : Vx += 0.1 / -= 0.1
    Home/End   : Vy += 0.1 / -= 0.1
    Insert/Del : Wz += 0.1 / -= 0.1
    F1         : reset to cmd_init
    P          : reset robot to default pose
"""

import copy
import json
import os
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
import onnxruntime as ort
import yaml
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# MuJoCo body ordering in n2_18dof.xml (matches MotionLoaderNingTracking):
#   0  base_link
#   1  L_arm_shoulder_pitch_Link
#   2  L_arm_shoulder_roll_Link
#   3  L_arm_shoulder_yaw_Link
#   4  L_arm_elbow_Link
#   5  L_arm_hand_Link
#   6  R_arm_shoulder_pitch_Link
#   7  R_arm_shoulder_roll_Link
#   8  R_arm_shoulder_yaw_Link
#   9  R_arm_elbow_Link
#  10  R_arm_hand_Link
#  11  L_leg_hip_yaw_link
#  12  L_leg_hip_roll_link
# 13  L_leg_hip_pitch_link
#  14  L_leg_knee_link
#  15  L_leg_ankle_link
#  16  R_leg_hip_yaw_link
#  17  R_leg_hip_roll_link
#  18  R_leg_hip_pitch_link
#  19  R_leg_knee_link
#  20  R_leg_ankle_link
#
# MotionLoaderNingTracking maps only the first 20 bodies (no R_arm_hand_Link).
# --------------------------------------------------------------------------- #

MOTION_LOADER_BODY_NAMES = [
    "base_link",
    "L_arm_shoulder_pitch_Link",
    "L_arm_shoulder_roll_Link",
    "L_arm_shoulder_yaw_Link",
    "L_arm_elbow_Link",
    "L_arm_hand_Link",
    "R_arm_shoulder_pitch_Link",
    "R_arm_shoulder_roll_Link",
    "R_arm_shoulder_yaw_Link",
    "R_arm_elbow_Link",
    "R_arm_hand_Link",
    "L_leg_hip_yaw_link",
    "L_leg_hip_roll_link",
    "L_leg_hip_pitch_link",
    "L_leg_knee_link",
    "L_leg_ankle_link",
    "R_leg_hip_yaw_link",
    "R_leg_hip_roll_link",
    "R_leg_hip_pitch_link",
    "R_leg_knee_link",
    "R_leg_ankle_link",
]
NUM_REFERENCE_BODIES = 20  # motion loader uses 20 bodies (excludes R_arm_hand_Link)


# --------------------------------------------------------------------------- #
# Keyboard command handler
# --------------------------------------------------------------------------- #

class cmd:
    def __init__(self, init_cmd):
        self.cmd = np.array(init_cmd, dtype=np.float32)
        self.reset_flag = False  # set True by keyboard to trigger reset

    def cmd_swtich(self, key_input):
        from pynput.keyboard import Key
        if key_input == Key.up:
            self.cmd[0] += 0.1
        elif key_input == Key.down:
            self.cmd[0] -= 0.1
        elif key_input == Key.home:
            self.cmd[1] += 0.1
        elif key_input == Key.end:
            self.cmd[1] -= 0.1
        elif key_input == Key.insert:
            self.cmd[2] += 0.1
        elif key_input == Key.delete:
            self.cmd[2] -= 0.1
        elif key_input == Key.f1:
            self.cmd[:] = 0.0
        elif hasattr(key_input, 'char') and key_input.char.lower() == 'p':
            self.reset_flag = True
        print(f"[cmd] Vx={self.cmd[0]:.1f}  Vy={self.cmd[1]:.1f}  Wz={self.cmd[2]:.1f}")


# --------------------------------------------------------------------------- #
# Motion loader (reads Walking_3.json like the IsaacGym version)
# --------------------------------------------------------------------------- #

class MotionLoaderNingTracking:
    """
    Replicates the key data-loading logic of MotionLoaderNingTracking from
    humanoid/amp_utils/motion_loader.py for use in the sim2sim loop.

    Original frame layout (111 dims, 0-indexed):
      0-2   : root_pos (3)
      3-6   : root_rot (4)  ← qx/qy/qz/qw in file
      7-28  : joint_pose (18)   ← indices 7..24 in full frame
      29-88 : tar_toe_pos_local (60)  ← indices 29..88
      89-91 : linear_vel (3)
      92-94 : angular_vel (3)
      95-112: joint_vel (18)
      113-114: contact_mask (2)

    After slicing to ROOT_ROT_END_IDX=7 (columns 7 onwards), the trajectory has
    104 dims (indices 0..103 in the sliced array):
      0-17  : joint_pose (18)   ← original idx 7..24
      18-77 : tar_toe_pos_local (60)
      78-80 : linear_vel (3)
      81-83 : angular_vel (3)
      84-101: joint_vel (18)
      102-103: contact_mask (2)
    """

    # Slice indices into the sliced trajectory (104-dim, 0-indexed from col 7)
    JP_START, JP_END = 0, 18       # joint_pose
    TT_START, TT_END = 18, 78      # tar_toe_pos_local
    LV_START, LV_END = 78, 81      # linear_vel
    AV_START, AV_END = 81, 84      # angular_vel
    JV_START, JV_END = 84, 102     # joint_vel
    CM_START, CM_END = 102, 104    # contact_mask

    GAIT_FREQ = 1.5  # ~1.5 Hz walking

    def __init__(self, motion_file):
        with open(motion_file) as f:
            mj = json.load(f)
        frames = np.array(mj["Frames"], dtype=np.float32)
        self.frame_duration = float(mj["FrameDuration"])
        self.num_frames = frames.shape[0]
        self.motion_len_s = (self.num_frames - 1) * self.frame_duration

        # Normalize quaternion: [qx,qy,qz,qw] → [qw,qx,qy,qz]
        for i in range(self.num_frames):
            qx, qy, qz, qw = frames[i, 3:7]
            frames[i, 3:7] = [qw, qx, qy, qz]

        # Slice from ROOT_ROT_END_IDX=7 (drop root_pos + root_rot)
        self.trajectory = frames[:, 7:]  # (num_frames, 104)

        self.ref_joint_pose = self.trajectory[:, self.JP_START:self.JP_END]
        self.ref_tar_toe    = self.trajectory[:, self.TT_START:self.TT_END]

        print(f"[MotionLoader] {motion_file}")
        print(f"  frames={self.num_frames}, frame_duration={self.frame_duration:.5f}s, "
              f"len={self.motion_len_s:.2f}s, trajectory_dims={self.trajectory.shape[1]}")
        print(f"  joint_pose shape: {self.ref_joint_pose.shape}")

    def get_frame_at_time(self, t):
        """Return (joint_pose[18], tar_toe_pos_local[60]) at time t (seconds)."""
        idx_f = t / self.frame_duration
        idx0 = int(idx_f) % self.num_frames
        idx1 = (idx0 + 1) % self.num_frames
        alpha = idx_f - int(idx_f)
        jp = (1 - alpha) * self.ref_joint_pose[idx0] + alpha * self.ref_joint_pose[idx1]
        tt = (1 - alpha) * self.ref_tar_toe[idx0]    + alpha * self.ref_tar_toe[idx1]
        return jp.astype(np.float32), tt.astype(np.float32)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def get_obs_from_mujoco(data):
    """Extract observation fields from MuJoCo data."""
    q = data.qpos.astype(np.float64)
    dq = data.qvel.astype(np.float64)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float64)
    r = R.from_quat(quat)
    omega = data.sensor("angular-velocity").data.astype(np.float64)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float64)
    return q, dq, quat, omega, gvec


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


# --------------------------------------------------------------------------- #
# Main sim2sim loop
# --------------------------------------------------------------------------- #

def run_mujoco_motion(cfg_path, headless=False):
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    def r(path):
        return path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    policy_path = r(config["policy_path"])
    xml_path    = r(config["xml_path"])

    sim_dt        = config["simulation_dt"]
    ctrl_decim    = config["control_decimation"]
    sim_duration  = config["simulation_duration"]
    kps           = np.array(config["kps"], dtype=np.float32)
    kds           = np.array(config["kds"], dtype=np.float32)
    default_angles= np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale = float(config["ang_vel_scale"])
    dof_pos_scale = float(config["dof_pos_scale"])
    dof_vel_scale = float(config["dof_vel_scale"])
    action_scale  = float(config["action_scale"])
    cmd_scale     = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions   = config["num_actions"]
    num_single    = config["num_single_obs"]
    frame_stack   = config["frame_stack"]
    num_obs       = config["num_obs"]
    motion_file   = r(config["motion_file"])
    motion_frame_duration = float(config.get("motion_frame_duration", 0.0334))
    motion_num_frames     = int(config.get("motion_num_frames", 588))
    cmd_init      = config.get("cmd_init", [0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ #
    # Load ONNX policy
    # ------------------------------------------------------------------ #
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    policy = ort.InferenceSession(policy_path, sess_opts, providers=["CPUExecutionProvider"])
    print(f"[Policy] Loaded ONNX: {policy_path}")

    # ------------------------------------------------------------------ #
    # Load MuJoCo model
    # ------------------------------------------------------------------ #
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = sim_dt
    data = mujoco.MjData(model)

    # Debug: print DOF info
    print(f"[MuJoCo] nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"[MuJoCo] qpos shape: {data.qpos.shape}, qvel shape: {data.qvel.shape}")
    print(f"[MuJoCo] root qpos[0:7]={data.qpos[:7]}")
    print(f"[MuJoCo] root qvel[0:6]={data.qvel[:6]}")
    print(f"[MuJoCo] joint qpos[7:] len={len(data.qpos[7:])}, joint qvel[6:] len={len(data.qvel[6:])}")

    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in MOTION_LOADER_BODY_NAMES]

    joint_ids = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(model.njnt)]
    print(f"[MuJoCo] total joints={model.njnt}: {joint_ids}")

    # ------------------------------------------------------------------ #
    # Load motion
    # ------------------------------------------------------------------ #
    motion = MotionLoaderNingTracking(motion_file)

    # ------------------------------------------------------------------ #
    # Initialise robot state
    # ------------------------------------------------------------------ #
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_geom_ids = [i for i in range(model.ngeom)
                     if model.geom_bodyid[i] == base_body_id]

    def reset_robot():
        """Reset robot to motion frame 0 (standing at start of walking cycle)."""
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.ctrl[:] = 0.0
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.75
        data.qpos[3] = 1.0
        data.qpos[4:7] = 0.0
        # Use motion frame 0: left foot elevated (swing phase), right foot planted
        data.qpos[7:] = motion.ref_joint_pose[0]
        mujoco.mj_step(model, data)
        hist_obs.clear()
        for _ in range(frame_stack):
            hist_obs.append(zero_obs.copy())
        prev_action[:] = 0.0
        action[:] = 0.0
        target_q[:] = motion.ref_joint_pose[0]
        print(f"[Reset] Robot reset to motion frame 0 at t={sim_time:.2f}s")

    # Initialise: use motion frame 0 joint pose (right foot planted, left foot swing)
    data.qpos[7:] = motion.ref_joint_pose[0]
    mujoco.mj_step(model, data)
    mujoco.mj_step(model, data)  # one more step for FK to settle

    if headless:
        viewer = None
    else:
        viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q  = np.zeros(num_actions, dtype=np.float64)
    prev_action = np.zeros(num_actions, dtype=np.float32)
    action    = np.zeros(num_actions, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Frame-stack history
    # ------------------------------------------------------------------ #
    hist_obs = deque()
    zero_obs = np.zeros([1, num_single], dtype=np.float32)
    for _ in range(frame_stack):
        hist_obs.append(zero_obs.copy())

    # ------------------------------------------------------------------ #
    # Tracking metrics
    # ------------------------------------------------------------------ #
    joint_pos_errors = []
    lfoot_forces = []
    rfoot_forces = []

    sim_time = 0.0
    gait_period = 1.0 / motion.GAIT_FREQ
    gait_phase = 0.0

    total_steps = int(sim_duration / sim_dt)
    print(f"[Sim] {total_steps} steps @ {1/sim_dt:.0f}Hz, ctrl @ {1/(sim_dt*ctrl_decim):.0f}Hz")

    fallen = False
    base_contact_reported = False

    for step in tqdm(range(total_steps), desc="Simulating motion..."):

        # -- handle keyboard reset -- #
        if command.reset_flag:
            command.reset_flag = False
            reset_robot()
            fallen = False
            base_contact_reported = False
            continue  # skip this step, next iteration starts fresh

        # -- read mujoco sensors -- #
        q, dq, quat, omega, gvec = get_obs_from_mujoco(data)
        joint_pos = q[7:].astype(np.float32)
        joint_vel = dq[6:].astype(np.float32)

        # -- fall detection: base z < 0.2 m -- #
        if not fallen and q[2] < 0.2:
            fallen = True
            print(f"\n[FALL DETECTED] t={sim_time:.2f}s  base_z={q[2]:.3f}m  "
                  f"pitch={np.degrees(2*np.arctan2(quat[3], quat[0])):.1f}deg  "
                  f"roll={np.degrees(2*np.arctan2(quat[2], quat[1])):.1f}deg")
            print(f"  Joint errors: {np.linalg.norm(joint_pos - default_angles):.3f} rad")
            print(f"  Base pos: {q[:3]}")
            print(f"  Root quat (qw,qx,qy,qz): {quat}")
            print(f"  Omega: {omega}")
            print(f"  Gravity vector: {gvec}")

        # -- base contact detection: check every step -- #
        if not base_contact_reported:
            for contact in data.contact[:data.ncon]:
                g1, g2 = contact.geom1, contact.geom2
                if g1 in base_geom_ids or g2 in base_geom_ids:
                    base_contact_reported = True
                    print(f"\n[BASE CONTACT] t={sim_time:.2f}s  "
                          f"base_z={q[2]:.3f}m  "
                          f"pitch={np.degrees(2*np.arctan2(quat[3], quat[0])):.1f}deg  "
                          f"roll={np.degrees(2*np.arctan2(quat[2], quat[1])):.1f}deg")
                    print(f"  contact geom1={g1} geom2={g2}  "
                          f"dist={contact.dist:.4f}m  "
                          f"pos={contact.pos}")
                    break

        # -- build observation every control_decimation steps -- #
        if step % ctrl_decim == 0:
            # Phase from sim_time relative to full motion length (matches IsaacGym MimicEnv):
            #   phase = (sim_time / motion_len_s) % 1.0
            gait_phase = (sim_time / motion.motion_len_s) % 1.0
            sin_phase = np.sin(2 * np.pi * gait_phase, dtype=np.float32)
            cos_phase = np.cos(2 * np.pi * gait_phase, dtype=np.float32)

            # Single-step obs (62 dims, matching IsaacGym N2MimicEnv.compute_observations):
            #   [0:2]   sin/cos gait phase (2)     ← from motion time
            #   [2:5]   ang_vel (3)                 ← base angular velocity
            #   [5:8]   gravity (3)               ← projected gravity in body frame
            #   [8:26]  joint_pos - default (18)   ← joint position error
            #   [26:44] joint_vel (18)             ← joint velocities
            #   [44:62] prev_action (18)          ← previous policy action
            obs = np.zeros([1, num_single], dtype=np.float32)
            obs[0, 0]   = sin_phase
            obs[0, 1]   = cos_phase
            obs[0, 2:5] = omega.astype(np.float32) * ang_vel_scale
            obs[0, 5:8] = gvec.astype(np.float32)
            obs[0, 8:26]= (joint_pos - default_angles) * dof_pos_scale
            obs[0, 26:44]= joint_vel * dof_vel_scale
            obs[0, 44:62]= prev_action

            hist_obs.append(obs)
            hist_obs.popleft()

            # Assemble frame-stacked input
            model_input = np.zeros([1, num_obs], dtype=np.float32)
            for i in range(frame_stack):
                model_input[0, i * num_single : (i + 1) * num_single] = hist_obs[i][0, :]

            # Policy inference
            action[:] = policy.run(None, {"policy_input": model_input})[0][0]
            prev_action[:] = action

            target_q = action * action_scale + default_angles

        # -- PD torque control -- #
        tau = pd_control(target_q.astype(np.float64), joint_pos.astype(np.float64),
                         kps.astype(np.float64), np.zeros(num_actions),
                         joint_vel.astype(np.float64), kds.astype(np.float64))
        data.ctrl = tau

        # -- log -- #
        if step % 500 == 0:
            ref_j, _ = motion.get_frame_at_time(sim_time)
            err = np.linalg.norm(joint_pos.astype(np.float64) - ref_j.astype(np.float64))
            lf = data.sensor("L_leg_foot_force").data[2]
            rf = data.sensor("R_leg_foot_force").data[2]
            print(f"  t={sim_time:.2f}s | phase={gait_phase:.2f} | "
                  f"base_z={q[2]:.3f}m | {'[FALLEN] ' if fallen else ''}"
                  f"joint_err={err:.3f} | Lf={lf:.1f}Nf | Rf={rf:.1f}Nf | "
                  f"action=[{action.min():.2f},{action.max():.2f}]")

        joint_pos_errors.append(float(np.linalg.norm(
            joint_pos.astype(np.float64) - motion.get_frame_at_time(sim_time)[0].astype(np.float64))))
        lfoot_forces.append(float(data.sensor("L_leg_foot_force").data[2]))
        rfoot_forces.append(float(data.sensor("R_leg_foot_force").data[2]))

        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.render()
        sim_time += sim_dt

    if viewer is not None:
        viewer.close()

    # ------------------------------------------------------------------ #
    # Plot results
    # ------------------------------------------------------------------ #
    ts = np.arange(len(joint_pos_errors)) * sim_dt
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(ts, joint_pos_errors, color="tab:blue")
    axes[0].set_ylabel("Joint pos error (rad)")
    axes[0].set_title("Joint Tracking Error over Time")
    axes[0].grid(True)

    axes[1].plot(ts, lfoot_forces, color="tab:green", label="Left")
    axes[1].plot(ts, rfoot_forces, color="tab:red",   label="Right")
    axes[1].set_ylabel("Vertical force (N)")
    axes[1].set_title("Foot Contact Forces")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(ts, lfoot_forces, color="tab:green", label="Left")
    axes[2].plot(ts, rfoot_forces, color="tab:red",   label="Right")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Vertical force (N)")
    axes[2].set_title("Foot Contact Forces (zoomed)")
    axes[2].set_xlim([0, min(5.0, sim_duration)])
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f"n2_walking3 sim2sim — {sim_duration:.0f}s, cmd=[{cmd_init[0]},{cmd_init[1]},{cmd_init[2]}]",
                 fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(policy_path), "sim2sim_motion_result.png")
    plt.savefig(out_png, dpi=150)
    print(f"[Plot] Saved to {out_png}")
    plt.show()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    from yaml import Loader as yaml_Loader

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="n2_motion.yaml")
    parser.add_argument("--headless", action="store_true", help="Run without GUI viewer")
    args = parser.parse_args()

    cfg_path = os.path.join(LEGGED_GYM_ROOT_DIR, "sim2sim", "configs", args.config_file)
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml_Loader)
    cmd_init = config.get("cmd_init", [0.0, 0.0, 0.0])

    command = cmd(cmd_init)
    if not args.headless:
        from pynput.keyboard import Listener
        listener = Listener(on_press=command.cmd_swtich)
        listener.start()

    run_mujoco_motion(cfg_path, headless=args.headless)
