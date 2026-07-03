# 导入操作系统相关功能
import os
# 导入系统相关功能
import sys
import argparse
# 从humanoid模块导入根目录路径
from humanoid import LEGGED_GYM_ROOT_DIR

# 导入Isaac Gym库
import isaacgym
# 导入所有环境相关模块
from humanoid.envs import *
# 导入工具函数和类
from humanoid.utils import get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

# 导入数值计算库
import numpy as np
# 导入PyTorch深度学习框架
import torch


# 全局播放控制变量
EXPORT_POLICY = False  # 是否导出策略（已被 --export_policy CLI 参数替代）
CONTROL_ROBOT = False  # 是否控制机器人
RECORD_FRAMES = False  # 是否录制帧
MOVE_CAMERA = False  # 是否移动相机


def play(args):
    """
    播放/测试函数：加载训练好的策略模型并在环境中运行以可视化结果

    参数:
        args: 命令行参数对象，包含运行所需的各种配置
    """
    if not hasattr(args, 'num_envs') or args.num_envs is None:
        args.num_envs = 1
    if not hasattr(args, 'num_steps') or args.num_steps is None:
        args.num_steps = 2000
    if not hasattr(args, 'episode_length_s') or args.episode_length_s is None:
        args.episode_length_s = 100
    if not hasattr(args, 'cmd_lin_vel_x') or args.cmd_lin_vel_x is None:
        args.cmd_lin_vel_x = 1.0
    if not hasattr(args, 'cmd_lin_vel_y') or args.cmd_lin_vel_y is None:
        args.cmd_lin_vel_y = 0.0
    if not hasattr(args, 'cmd_ang_vel_yaw') or args.cmd_ang_vel_yaw is None:
        args.cmd_ang_vel_yaw = 0.0
    if not hasattr(args, 'save_path'):
        args.save_path = None
    if not hasattr(args, 'plot'):
        args.plot = False
    if not hasattr(args, 'verbose'):
        args.verbose = False
    if not hasattr(args, 'video_dir') or args.video_dir is None:
        args.video_dir = None

    # 根据 --video_dir 决定是否录制帧
    video_dir = args.video_dir
    record_frames = video_dir is not None
    if record_frames:
        os.makedirs(video_dir, exist_ok=True)
        print(f"[video] Recording frames to: {video_dir}")

    # 获取环境和训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 重写一些测试参数
    env_cfg.env.num_envs = args.num_envs  # 通过命令行参数控制
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**10  # 设置GPU接触对的最大数量
    env_cfg.terrain.mesh_type = 'plane'  # 设置地形类型为平面
    env_cfg.terrain.num_rows = 20  # 设置地形行数
    env_cfg.terrain.num_cols = 10  # 设置地形列数
    env_cfg.terrain.curriculum = False  # 关闭课程学习
    env_cfg.env.episode_length_s = args.episode_length_s  # 通过命令行参数控制
    env_cfg.env.test = True

    # Domain randomization: 默认关闭保守模式，与训练环境保持一致时启用
    if getattr(args, 'domain_rand', False):
        # 与训练一致: 启用观测噪声 + 物理随机化
        env_cfg.noise.add_noise = True
        env_cfg.domain_rand.randomize_gains = True
        env_cfg.domain_rand.randomize_motor_strength = True
        env_cfg.domain_rand.randomize_base_mass = True
        env_cfg.domain_rand.randomize_com_displacement = True
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.randomize_restitution = True
        env_cfg.domain_rand.push_robots = False  # 保持关闭，避免测试时意外干扰
        env_cfg.domain_rand.disturbance = False
        env_cfg.domain_rand.disturbance_probabilities = 0.0
        env_cfg.domain_rand.push_force_range = [0.0, 0.0]
        env_cfg.domain_rand.push_torque_range = [0.0, 0.0]
    else:
        # 保守模式: 关闭所有随机化，便于观察裸策略行为
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_gains = False
        env_cfg.domain_rand.randomize_motor_strength = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_com_displacement = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_restitution = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.disturbance = False
        env_cfg.domain_rand.disturbance_probabilities = 0.0
        env_cfg.domain_rand.push_force_range = [0.0, 0.0]
        env_cfg.domain_rand.push_torque_range = [0.0, 0.0]

    # 如果控制机器人标志为真，则进一步调整参数
    if CONTROL_ROBOT:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)  # 确保环境数量不超过1
        env_cfg.env.episode_length_s = 100  # 设置episode时长
        env_cfg.commands.resampling_time = [1000, 1001]  # 设置命令重采样时间
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]  # 设置x方向线速度范围
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]  # 设置y方向线速度范围
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]  # 设置偏航角速度范围

    # 准备环境
    # env: 环境对象，用于模拟和交互
    # _: 忽略第二个返回值
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 获取初始观测值
    obs = env.get_observations()

    # 加载策略
    train_cfg.runner.resume = True  # 设置为恢复模式
    # 创建算法运行器实例
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # 获取推理策略
    policy = ppo_runner.get_inference_policy(device=env.device)

    # 将策略导出为JIT模块（用于C++中运行）
    if getattr(args, 'export_policy', False):
        # 构建导出路径：sim2sim/policy/{export_name}/
        export_name = getattr(args, 'export_name', None) or 'policy'
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'sim2sim', 'policy', export_name)
        os.makedirs(path, exist_ok=True)
        # 使用自定义文件名 xx.pt / xx.onnx
        export_policy_as_jit(ppo_runner.alg.policy, path, ppo_runner.obs_normalizer,
                             filename=f'{export_name}.pt')
        export_policy_as_onnx(ppo_runner.alg.policy, path, ppo_runner.obs_normalizer,
                              filename=f'{export_name}.onnx')
        # 额外导出观测归一化参数（均值/标准差），供其他仿真平台使用
        # 仅当 normalizer 不是 Identity（即训练时启用了观测归一化）才保存
        norm = ppo_runner.obs_normalizer
        norm_type = type(norm).__name__
        if norm_type != 'Identity':
            np.save(os.path.join(path, 'obs_mean.npy'), norm.mean.cpu().numpy())
            np.save(os.path.join(path, 'obs_std.npy'),  norm.std.cpu().numpy())
            print(f"[export] Saved obs_mean.npy and obs_std.npy")
        else:
            print(f"[export] Normalizer is Identity (no normalization used in training) — skipping obs_mean/std export.")
        print(f'Exported policy to: {path}')

    # 创建日志记录器
    logger = Logger(env)
    robot_index = 0  # 用于日志记录的机器人索引
    joint_index = 1  # 用于日志记录的关节索引
    stop_state_log = 100  # 开始绘制状态前的步数
    stop_rew_log = env.max_episode_length + 1  # 开始打印平均奖励前的步数
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)  # 相机位置
    camera_vel = np.array([1., 1., 0.])  # 相机速度
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)  # 相机方向
    img_idx = 0  # 图像索引

    # ============ 固定相机（num_envs==1 专用）============
    # 根据机器人实际朝向计算"右方"，将相机锁定到机器人右上方。
    # IsaacGym root_quat 顺序为 (x, y, z, w)，世界系：+x 前，+y 左，+z 上。
    FIX_CAMERA = (args.num_envs == 1)
    if FIX_CAMERA:
        init_root = env.root_states[robot_index, :3].detach().cpu().numpy().copy()
        qx, qy, qz, qw = env.root_states[robot_index, 3:7].detach().cpu().numpy()
        # 由四元数提取机器人本地坐标轴在世界系中的方向
        # forward (本地 +x) -> 世界 fwd
        fwd_x = 1 - 2 * (qy * qy + qz * qz)
        fwd_y = 2 * (qx * qy + qz * qw)
        # right  = forward × world_up(0,0,1)，等价于本地 +y 在世界系的方向
        right_x = 2 * (qx * qy - qz * qw)
        right_y = 1 - 2 * (qx * qx + qz * qz)
        norm = np.sqrt(right_x * right_x + right_y * right_y) + 1e-8
        right_x /= norm
        right_y /= norm
        # 相机距离（米）：向右 3.0、向上 0；前向偏移 0.0（正右方）
        cam_right_dist = -3.0
        cam_up = 0
        cam_fwd = 0.0  # 0 = 正右方；>0 则更偏前
        fixed_cam_pos = np.array([
            init_root[0] + cam_right_dist * right_x + cam_fwd * fwd_x,
            init_root[1] + cam_right_dist * right_y + cam_fwd * fwd_y,
            init_root[2] + cam_up,
        ], dtype=np.float64)
        fixed_lookat = init_root + np.array([0.0, 0.0, 0.6], dtype=np.float64)  # 看机器人的躯干中部
        env.set_camera(fixed_cam_pos, fixed_lookat)

    # ============ 数据记录 ============
    num_dofs = env.num_actions
    has_motion = hasattr(env, 'motion_loader') and env.motion_loader is not None
    ml = env.motion_loader if has_motion else None
    motion_time = 0.0  # 累计播放时间，对齐 reference motion
    frame_duration = getattr(ml, 'time_between_frames', 1.0/60.0) if has_motion else 1.0/60.0

    recorded_steps = {
        't': [],
        'dof_pos': [],
        'dof_pos_target': [],
        'dof_vel': [],
        'base_lin_vel': [],
        'base_ang_vel': [],
        'root_pos': [],
        'root_quat': [],
        'commands': [],
        'contact_forces': [],
        'reward': [],
        'reference_dof_pos': [],
        'reference_root_pos': [],
    }

    num_steps = args.num_steps
    print(f"\n=== Play Config ===")
    print(f"Task: {args.task}")
    print(f"Load run: {args.load_run}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num envs: {args.num_envs}")
    print(f"Num steps: {num_steps}")
    print(f"Episode length: {env.max_episode_length} steps ({args.episode_length_s}s)")
    print(f"Motion loader: {'Yes' if has_motion else 'No'}")
    print(f"Commands: Vx={args.cmd_lin_vel_x}, Vy={args.cmd_lin_vel_y}, Wz={args.cmd_ang_vel_yaw}")
    print(f"Save path: {args.save_path}")
    print(f"====================\n")

    # 主循环：运行指定次数的episode
    env.commands[:, 0] = args.cmd_lin_vel_x
    env.commands[:, 1] = args.cmd_lin_vel_y
    env.commands[:, 2] = args.cmd_ang_vel_yaw

    reset_count = 0
    for i in range(num_steps):
        # 获取策略动作
        actions = policy(obs.detach())
        # 执行动作
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        t = i * env.dt
        recorded_steps['t'].append(t)
        recorded_steps['dof_pos'].append(env.dof_pos[robot_index].cpu().numpy().copy())
        recorded_steps['dof_pos_target'].append(
            (env.dof_pos[robot_index].cpu().numpy()
             + actions[robot_index].detach().cpu().numpy() * env.cfg.control.action_scale).copy()
        )
        recorded_steps['dof_vel'].append(env.dof_vel[robot_index].cpu().numpy().copy())
        recorded_steps['base_lin_vel'].append(env.base_lin_vel[robot_index].cpu().numpy().copy())
        recorded_steps['base_ang_vel'].append(env.base_ang_vel[robot_index].cpu().numpy().copy())
        recorded_steps['root_pos'].append(env.root_states[robot_index, :3].cpu().numpy().copy())
        recorded_steps['root_quat'].append(env.root_states[robot_index, 3:7].cpu().numpy().copy())
        recorded_steps['commands'].append(env.commands[robot_index].cpu().numpy().copy())
        recorded_steps['contact_forces'].append(env.contact_forces[robot_index].cpu().numpy().copy())
        recorded_steps['reward'].append(rews[robot_index].item())

        if has_motion:
            from humanoid.amp_utils.motion_loader import MotionLoaderNingTracking as MLNT
            # Get reference pose at current motion time
            ref_frame = ml.get_full_frame_at_time_batch(np.array([0]), np.array([motion_time]))
            # Joint poses: after root pos(3) + root rot(4) = 7
            ref_dof = ref_frame[0, MLNT.JOINT_POSE_START_IDX:MLNT.JOINT_POSE_END_IDX].cpu().numpy()
            recorded_steps['reference_dof_pos'].append(ref_dof.copy())
            recorded_steps['reference_root_pos'].append(ref_frame[0, :3].cpu().numpy())
            motion_time += frame_duration
            if motion_time > ml.trajectory_lens[0]:
                motion_time = 0.0  # Loop motion
        else:
            recorded_steps['reference_dof_pos'].append(np.zeros(num_dofs))
            recorded_steps['reference_root_pos'].append(np.zeros(3))

        # 原有的日志记录（仅前 stop_state_log 步）
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )

        # 录制帧
        if record_frames:
            filename = os.path.join(video_dir, f"frame_{img_idx:06d}.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1

        # 移动相机（MOVE_CAMERA=True 时由脚本驱动相机平移；
        # FIX_CAMERA 仅在循环开始前 set 一次作为初始视角，
        # 此后不覆盖 viewer，让 IsaacGym viewer 的鼠标交互生效）
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # 奖励记录
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

        # 重置计数
        if dones[robot_index].item() > 0.5:
            reset_count += 1
            if args.verbose:
                ep_rew = sum(recorded_steps['reward'][-env.max_episode_length:])
                print(f"  [Step {i}] Reset @ t={t:.2f}s, ep_rew={ep_rew:.2f}")

    # ============ 结果汇总 ============
    total_rew = sum(recorded_steps['reward'])
    mean_rew = np.mean(recorded_steps['reward'])
    print(f"\n--- Play Results ---")
    print(f"Total steps: {num_steps} | Resets: {reset_count}")
    print(f"Total reward: {total_rew:.2f} | Mean/step: {mean_rew:.4f}")

    # ============ 保存数据 ============
    if args.save_path:
        import json
        save_dict = {}
        for k, v in recorded_steps.items():
            if k in ['dof_pos', 'dof_pos_target', 'dof_vel', 'base_lin_vel',
                     'base_ang_vel', 'root_pos', 'root_quat', 'commands',
                     'contact_forces', 'reference_dof_pos', 'reference_root_pos']:
                save_dict[k] = np.array(v).tolist()
            else:
                save_dict[k] = v
        with open(args.save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        print(f"Data saved: {args.save_path}")

    # ============ 绘图 ============
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            t = recorded_steps['t']
            dof_names = env.dof_names

            fig, axes = plt.subplots(6, 1, figsize=(18, 28), sharex=True)

            ax = axes[0]
            for j in range(min(num_dofs, 9)):
                ax.plot(t, [d[j] for d in recorded_steps['dof_pos']],
                        label=f'{dof_names[j]}_actual', alpha=0.8)
                ref = [d[j] for d in recorded_steps['reference_dof_pos']]
                if has_motion and any(r != 0 for r in ref):
                    ax.plot(t, ref, '--', label=f'{dof_names[j]}_ref', alpha=0.5)
            ax.set_ylabel('Joint Pos (rad)')
            ax.set_title('Joint Positions (actual vs reference)')
            ax.legend(loc='upper right', fontsize=5, ncol=2)
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            for j, lbl in enumerate(['Rx', 'Ry', 'Rz']):
                ax.plot(t, [d[j] for d in recorded_steps['root_pos']], label=f'{lbl}_actual', alpha=0.8)
                if has_motion:
                    ref = [d[j] for d in recorded_steps['reference_root_pos']]
                    if any(r != 0 for r in ref):
                        ax.plot(t, ref, '--', label=f'{lbl}_ref', alpha=0.5)
            ax.set_ylabel('Root Pos (m)')
            ax.set_title('Root Position (actual vs reference)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            ax = axes[2]
            for j, lbl in enumerate(['Vx', 'Vy', 'Vz']):
                ax.plot(t, [v[j] for v in recorded_steps['base_lin_vel']], label=lbl)
            ax.set_ylabel('Lin Vel (m/s)')
            ax.set_title('Base Linear Velocity')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            ax = axes[3]
            for j, lbl in enumerate(['Wx', 'Wy', 'Wz']):
                ax.plot(t, [v[j] for v in recorded_steps['base_ang_vel']], label=lbl)
            ax.set_ylabel('Ang Vel (rad/s)')
            ax.set_title('Base Angular Velocity')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            ax = axes[4]
            cf = np.array(recorded_steps['contact_forces'])
            for j in range(cf.shape[1]):
                ax.plot(t, cf[:, j], label=f'foot_{j}')
            ax.set_ylabel('Contact Force Z (N)')
            ax.set_title('Contact Forces')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            ax = axes[5]
            ax.plot(t, recorded_steps['reward'], color='green', alpha=0.8)
            ax.set_ylabel('Reward')
            ax.set_xlabel('Time (s)')
            ax.set_title('Reward per Step')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = args.save_path.replace('.json', '.png') if args.save_path else 'play_data.png'
            plt.savefig(plot_path, dpi=120)
            print(f"Plot saved: {plot_path}")
        except Exception as e:
            print(f"Plot failed: {e}")


# 程序入口点
if __name__ == '__main__':
    # 从 sys.argv 中分离已知参数（get_args 用）和额外参数（play.py 专用）
    known_flag_set = {
        '--sim_device', '--pipeline', '--graphics_device_id',
        '--flex', '--physx', '--num_threads', '--subscenes', '--slices',
        '--task', '--resume', '--experiment_name', '--run_name',
        '--load_run', '--checkpoint', '--headless', '--horovod',
        '--rl_device', '--num_envs', '--seed', '--max_iterations',
    }
    base_argv = ['play.py']
    play_argv = []  # 收集 play.py 专用参数（get_args 不认识的）
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--'):
            if arg in known_flag_set:
                base_argv.append(arg)
                i += 1
                if i < len(sys.argv) and not sys.argv[i].startswith('--'):
                    base_argv.append(sys.argv[i])
                    i += 1
            else:
                # 收集 play.py 专用参数（避免被跳过）
                play_argv.append(arg)
                i += 1
                if i < len(sys.argv) and not sys.argv[i].startswith('--'):
                    play_argv.append(sys.argv[i])
                    i += 1
        elif not arg.startswith('-'):
            base_argv.append(arg)
            i += 1
        else:
            i += 1

    original_argv = sys.argv
    sys.argv = base_argv
    args = get_args()
    sys.argv = original_argv

    # 解析 play.py 专用参数
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--num_steps', type=int, default=2000)
    p.add_argument('--episode_length_s', type=int, default=100)
    p.add_argument('--cmd_lin_vel_x', type=float, default=1.0)
    p.add_argument('--cmd_lin_vel_y', type=float, default=0.0)
    p.add_argument('--cmd_ang_vel_yaw', type=float, default=0.0)
    p.add_argument('--save_path', type=str, default=None)
    p.add_argument('--plot', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--domain_rand', action='store_true', help='Enable domain randomization (noise + physics randomization) to match training conditions.')
    p.add_argument('--video_dir', type=str, default=None, help='Directory to save video frames and generate video. If set, enables frame recording.')
    p.add_argument('--export_policy', action='store_true',
                   help='Export the loaded policy as TorchScript (.pt) and ONNX (.onnx) after loading. '
                        'Also saves obs_mean.npy / obs_std.npy for use in other simulators.')
    p.add_argument('--export_name', type=str, default=None,
                   help='Custom name for the exported policy. Files are written to '
                        '{LEGGED_GYM_ROOT_DIR}/sim2sim/policy/{export_name}/ as '
                        '{export_name}.pt and {export_name}.onnx. Defaults to "policy".')
    parsed = p.parse_args(play_argv if play_argv else [])

    for attr in ['num_steps', 'episode_length_s', 'cmd_lin_vel_x', 'cmd_lin_vel_y',
                 'cmd_ang_vel_yaw', 'save_path', 'plot', 'verbose', 'domain_rand', 'video_dir',
                 'export_policy', 'export_name']:
        val = getattr(parsed, attr)
        if val is not None:
            setattr(args, attr, val)

    play(args)

    # ============ 生成视频 ============
    if parsed.video_dir:
        video_dir = parsed.video_dir
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.png')])
        if not frame_files:
            print(f"[video] No frames found in {video_dir}, skipping video generation.")
        else:
            video_path = os.path.join(video_dir, 'video.mp4')
            print(f"[video] Generating video from {len(frame_files)} frames ...")
            try:
                import imageio
                writer = imageio.get_writer(
                    video_path,
                    fps=30,
                    codec='libx264',
                    quality=8,
                    pixelformat='yuv420p',
                )
                for frame_file in frame_files:
                    frame = imageio.imread(os.path.join(video_dir, frame_file))
                    writer.append_data(frame)
                writer.close()
                print(f"[video] Video saved: {video_path}")
            except Exception as e:
                print(f"[video] Failed to generate video: {e}")