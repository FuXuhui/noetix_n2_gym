import torch.nn.functional as F
import math
import copy
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
import torch
from pynput.keyboard import Listener, Key, KeyCode
import yaml
import onnxruntime as ort

import matplotlib.pyplot as plt

class cmd:
    def __init__(self):
        self.cmd = np.array([0., 0., 0.], dtype=np.float32)
        self._keys_pressed = set()

    def on_press(self, key_input):
        if isinstance(key_input, KeyCode):
            k = key_input.char
        else:
            return  # skip non-char keys in this handler
        if k == '8':       # Forward
            self._keys_pressed.add('8')
        elif k == '2':       # Backward
            self._keys_pressed.add('2')
        elif k == '4':       # Turn left
            self._keys_pressed.add('4')
        elif k == '6':       # Turn right
            self._keys_pressed.add('6')
        elif k == '5':       # Stop all
            self._keys_pressed.discard('8')
            self._keys_pressed.discard('2')
            self._keys_pressed.discard('4')
            self._keys_pressed.discard('6')
        self._apply_keys()

    def on_release(self, key_input):
        if isinstance(key_input, KeyCode):
            k = key_input.char
        else:
            return
        self._keys_pressed.discard(k)
        self._apply_keys()

    def _apply_keys(self):
        forward  = 1.0 if '8' in self._keys_pressed else 0.0
        backward = 1.0 if '2' in self._keys_pressed else 0.0
        turn_l   = 1.0 if '4' in self._keys_pressed else 0.0
        turn_r   = 1.0 if '6' in self._keys_pressed else 0.0
        self.cmd[0] = forward - backward   # forward/back
        self.cmd[2] = turn_l - turn_r      # yaw
        self.cmd[1] = 0.0                 # lateral (not used)

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """

    with open(f"{LEGGED_GYM_ROOT_DIR}/sim2sim/configs/{cfg}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_single_obs = config["num_single_obs"]
        frame_stack = config["frame_stack"]

        # obs_format: "sim2sim_39d" (default) — policy_walk
        #            "sdk_40d"           — policy_user (with phase + euler)
        obs_format = config.get("obs_format", "sim2sim_39d")
        cycle_time = config.get("cycle_time", 1.0)
        lin_vel_scale = config.get("lin_vel_scale", float(cmd_scale[0]))
        ang_vel_yaw_scale = config.get("ang_vel_yaw_scale", float(cmd_scale[2]))
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = simulation_dt
    data = mujoco.MjData(model)

    # load policy — auto-detect ONNX vs TorchScript
    policy_path_ext = policy_path.split('.')[-1].lower()
    if policy_path_ext == 'onnx':
        print(f"Loading ONNX policy: {policy_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        policy = ort.InferenceSession(policy_path, sess_options, providers=['CPUExecutionProvider'])
        policy_is_onnx = True
    else:
        print(f"Loading TorchScript policy: {policy_path}")
        policy = torch.jit.load(policy_path)
        policy_is_onnx = False

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print("joint_names:", joint_names)
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print("actuator_names:", actuator_names)

    defaut_dof_pos = default_angles
    data.qpos[7:] = defaut_dof_pos

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((num_actions), dtype=np.double)
    action = np.zeros((num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(frame_stack):
        hist_obs.append(np.zeros([1, num_single_obs], dtype=np.double))

    count_lowlevel = 0
    L_foot_force_list = []
    R_foot_force_list = []
    phase_ = 0.0

    print(f"obs_format={obs_format}  cycle_time={cycle_time}  num_single_obs={num_single_obs}  frame_stack={frame_stack}")

    for _ in tqdm(range(int(simulation_duration / simulation_dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-num_actions:]
        dq = dq[-num_actions:]

        # advance phase_ on every control step
        if count_lowlevel % control_decimation == 0:
            phase_ += simulation_dt * control_decimation

        if count_lowlevel % control_decimation == 0:
            obs = np.zeros([1, num_single_obs], dtype=np.float32)

            if obs_format == "sdk_40d":
                # SDK 40-D format (matches noetix_sdk_release computeObservation)
                # [0:2]=phase, [2]=vel_x*lin_vel_scale, [3]=vel_y*lin_vel_scale,
                # [4]=vel_yaw*ang_vel_yaw_scale, [5:8]=omega, [8]=baseEulerX(roll), [9]=baseEulerY(pitch)
                # [10:20]=jointError, [20:30]=jointVel, [30:40]=prevAction
                phase = phase_ / cycle_time
                quat_wxyz = quat                              # mujoco returns [w,x,y,z]
                quat_xyzw = np.concatenate([quat_wxyz[1:], [quat_wxyz[0]]])
                euler_xyz = R.from_quat(quat_xyzw).as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
                obs[0, 0] = np.sin(2 * np.pi * phase)
                obs[0, 1] = np.cos(2 * np.pi * phase)
                obs[0, 2] = command.cmd[0] * lin_vel_scale
                obs[0, 3] = command.cmd[1] * lin_vel_scale
                obs[0, 4] = command.cmd[2] * ang_vel_yaw_scale
                obs[0, 5:8] = omega * ang_vel_scale
                obs[0, 8] = euler_xyz[0]                      # base roll
                obs[0, 9] = euler_xyz[1]                      # base pitch
                obs[0, 10:10 + num_actions] = (q - defaut_dof_pos) * dof_pos_scale
                obs[0, 10 + num_actions:10 + num_actions * 2] = dq * dof_vel_scale
                obs[0, 10 + num_actions * 2:10 + num_actions * 3] = action
            else:
                # sim2sim 39-D format (policy_walk)
                obs[0, :3] = command.cmd * cmd_scale
                obs[0, 3:6] = omega * ang_vel_scale
                obs[0, 6:9] = gvec[:3]
                obs[0, 9:9 + num_actions] = (q - defaut_dof_pos) * dof_pos_scale
                obs[0, 9 + num_actions:9 + num_actions * 2] = dq * dof_vel_scale
                obs[0, 9 + num_actions * 2:9 + num_actions * 3] = action

            hist_obs.append(obs)
            hist_obs.popleft()

            model_input = np.zeros([1, num_obs], dtype=np.float32)
            for i in range(frame_stack):
                model_input[0, i * num_single_obs : (i + 1) * num_single_obs] = hist_obs[i][0, :]
            policy_input = torch.tensor(model_input) if not policy_is_onnx else model_input

            if policy_is_onnx:
                action[:] = policy.run(None, {'policy_input': policy_input})[0][0]
            else:
                action[:] = policy(policy_input)[0].detach().numpy()

            target_q = (action * action_scale) + defaut_dof_pos
        
        L_leg_foot_force = data.sensor('L_leg_foot_force')
        R_leg_foot_force = data.sensor('R_leg_foot_force')

        if _ % 200 == 0:
            print(f"[{int(_*simulation_dt)}s] vel_x={v[0]:.3f}  cmd_fwd={command.cmd[0]:.2f}  cmd_yaw={command.cmd[2]:.2f}")

        L_foot_force_list.append(copy.copy(L_leg_foot_force.data[2]))
        R_foot_force_list.append(copy.copy(R_leg_foot_force.data[2]))

        target_dq = np.zeros((num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, kps,
                        target_dq, dq, kds)  # Calc torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1


    viewer.close()


if __name__ == '__main__':
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="n2_10dof.yaml", help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/sim2sim/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_single_obs = config["num_single_obs"]
        frame_stack = config["frame_stack"]
    
    command = cmd()
    if "cmd_init" in config:
        command.cmd = np.array(config["cmd_init"], dtype=np.float32)
    listener = Listener(on_press=command.on_press, on_release=command.on_release)
    listener.start()
    run_mujoco(config_file)
