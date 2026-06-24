from humanoid.envs.n2.n2_env import N2Env

import time
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch
import torch

from humanoid.utils.helpers import class_to_dict

from humanoid.envs.n2.n2_mimic_config import N2MimicCfg
from humanoid.utils.isaacgym_utils import get_euler_xyz_tensor
from humanoid.amp_utils.motion_loader import *


class N2MimicEnv(N2Env):
    def _create_envs(self):
        super()._create_envs()
        key_names = []
        for name in self.cfg.asset.key_name:
            key_names.extend([s for s in self.body_names if name in s])
        self.key_indices = torch.zeros(len(key_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(key_names)):
            self.key_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], key_names[i])
        
        upper_body_names = []
        for name in self.cfg.asset.upper_body_name:
            upper_body_names.extend([s for s in self.body_names if name in s])
        self.upper_body_indices = torch.zeros(len(upper_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(upper_body_names)):
            self.upper_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], upper_body_names[i])
    
        lower_body_names = []
        for name in self.cfg.asset.lower_body_name:
            lower_body_names.extend([s for s in self.body_names if name in s])
        self.lower_body_indices = torch.zeros(len(lower_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(lower_body_names)):
            self.lower_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], lower_body_names[i])

        # Clamp all body indices to valid range for ref_key_pos (MotionLoaderNingTracking has 20 bodies).
        # URDF has 21 bodies, so indices >= 20 must be clamped to 19.
        max_ref_idx = 19
        self.upper_body_indices = torch.clamp(self.upper_body_indices, 0, max_ref_idx)
        self.lower_body_indices = torch.clamp(self.lower_body_indices, 0, max_ref_idx)
        self.feet_indices = torch.clamp(self.feet_indices, 0, max_ref_idx)
        self.key_indices = torch.clamp(self.key_indices, 0, max_ref_idx)

    def _get_noise_scale_vec(self, cfg):
        if self.cfg.env.frame_stack is not None:
            noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        else:
            noise_vec = torch.zeros_like(self.obs_buf[0])

        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:2] = 0.  # commands
        noise_vec[2:5] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[5:8] = noise_scales.gravity * noise_level
        noise_vec[8:8+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[8+self.num_actions:8+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[8+2*self.num_actions:8+3*self.num_actions] = 0. # previous actions

        return noise_vec

    def update_key_pos_state(self):
        self.key_pos_state = self.rigid_body_states_view[:, self.key_indices, :]
        self.key_pos = self.key_pos_state[:, :, :3]

    def update_body_pos_state(self):
        self.upper_body_pos_state = self.rigid_body_states_view[:, self.upper_body_indices, :]
        self.upper_body_pos = self.upper_body_pos_state[:, :, :3]

        self.lower_body_pos_state = self.rigid_body_states_view[:, self.lower_body_indices, :]
        self.lower_body_pos = self.lower_body_pos_state[:, :, :3]

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.update_key_pos_state()
        self.update_body_pos_state()

    def _init_buffers(self):
        super()._init_buffers()
        self.cfg: N2MimicCfg
        self.estimator = None # assigned in runner
        self._init_adaptive_sigma()
        self.contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False) 
        # termination
        if self.cfg.termination.terminate_when_motion_far and self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum:
            self.terminate_when_motion_far_threshold = self.cfg.termination.termination_curriculum.terminate_when_motion_far_initial_threshold
        else:
            self.terminate_when_motion_far_threshold = self.cfg.termination.scales.termination_motion_far_threshold
        # get key pos state
        self.update_key_pos_state()
        # for reward penalty curriculum
        self.average_episode_length = 0.  # num_compute_average_epl last termination episode length
        self.current_mean_episode_length = 0.  # per-iteration mean, set by runner before curriculum update
        self.last_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.num_compute_average_epl = self.cfg.rewards.num_compute_average_epl
        # 过渡阶段状态缓冲区：0=站立稳定, 1=启动过渡, 2=正常motion跟踪
        self.transition_phase_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.transition_timer_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        # load motion components
        self.reference_motion_file = self.cfg.motion_loader.reference_motion_file
        self.reference_observation_horizon = self.cfg.motion_loader.reference_observation_horizon
        self.num_preload_transitions = self.cfg.motion_loader.num_preload_transitions
        self.motion_loader_class = eval(self.cfg.motion_loader.motion_loader_name)
        self.motion_loader: MotionLoaderNing = self.motion_loader_class(
            device=self.device,
            time_between_frames=self.dt,
            reference_observation_horizon=self.reference_observation_horizon,
            num_preload_transitions=self.num_preload_transitions,
            motion_files=self.reference_motion_file
        )
        self.motion_lenth = self.motion_loader.trajectory_lens[0]
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)

        # 合并 sigma_overrides 到 reward_tracking_sigma
        if hasattr(self.cfg.rewards, 'sigma_overrides'):
            for key, val in self.cfg.rewards.sigma_overrides.__dict__.items():
                if not key.startswith('_') and key in self.cfg.rewards.reward_tracking_sigma:
                    self.cfg.rewards.reward_tracking_sigma[key] = val

        self.compute_ref_state()
    
    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        if self.cfg.rewards.use_vec_reward:
            num_rew_fn = len(self.reward_functions)
            self.rew_buf = torch.zeros(self.num_envs, num_rew_fn, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.use_reward_penalty_curriculum = self.cfg.rewards.reward_penalty_curriculum
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = self.cfg.rewards.reward_initial_penalty_scale
    
    @property
    def num_rew_fn(self):
        return len(self.reward_functions) if self.cfg.rewards.use_vec_reward else 1
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # dynamic randomization
        # if not self.cfg.env.test:
        # delay = torch.rand((self.num_envs, 1), device=self.device) * 0.5
        # # delay = 0.2
        # actions = (1 - delay) * actions + delay * self.actions
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs, _ = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, \
            termination_priveleged_obs
    
    def  _get_phase(self):
        current_time = self.episode_length_buf * self.dt + self.motion_start_times
        phase = (current_time / self.motion_lenth) % 1.
        return phase
    
    def compute_ref_state(self):
        """计算参考状态。过渡阶段覆盖为站立/启动参考。"""
        phase = self._get_phase()
        motion_times = (phase * self.motion_lenth).clamp(min=0.0, max=self.motion_lenth-self.dt).cpu().numpy()
        frames = self.motion_loader.get_full_frame_at_time_batch(np.array([0] * self.num_envs), motion_times)

        # 获取参考线速度和角速度
        ref_lin_vel = self.motion_loader_class.get_linear_vel_batch(frames)
        self.ref_lin_vel = quat_rotate_inverse(self.base_quat, ref_lin_vel)

        ref_ang_vel = self.motion_loader_class.get_angular_vel_batch(frames)
        self.ref_ang_vel = quat_rotate_inverse(self.base_quat, ref_ang_vel)

        # 获取参考关节位置、速度和关键点位置
        self.ref_dof_pos = self.motion_loader_class.get_joint_pose_batch(frames)
        self.ref_dof_vel = self.motion_loader_class.get_joint_vel_batch(frames)
        self.ref_key_pos = self.motion_loader_class.get_tar_toe_pos_local_batch(frames)
        self.ref_contact_mask = self.motion_loader_class.get_contact_mask_batch(frames)

        # 过渡阶段覆盖参考状态
        self._override_ref_for_transition()

        # Pre-compute truncated key_pos (only first tar_toe_pos_local_num bodies)
        # ref_key_pos: (N, 60) = (N, 20*3), key_pos: (N, 21, 3) on GPU
        root_pos = self.root_states[:, :3]
        key_pos_full = (self.key_pos - root_pos.unsqueeze(1))
        num_ref_bodies = self.ref_key_pos.shape[-1] // 3  # 20
        self.key_pos_trunc = key_pos_full[:, :num_ref_bodies, :]

    def _override_ref_for_transition(self):
        """覆盖过渡阶段的参考状态：站立阶段用 URDF 默认，启动阶段平滑混合。"""
        if not hasattr(self, 'transition_phase_buf'):
            return

        stand_mask = self.transition_phase_buf == 0
        startup_mask = self.transition_phase_buf == 1
        default_dof = self.default_dof_pos  # [1, num_dof], broadcast-compatible with [N, num_dof]

        # 站立稳定阶段：关节位置用 default，速度=0，关键点位置用 motion frame 0 但速度=0
        if stand_mask.any():
            self.ref_dof_pos[stand_mask] = default_dof
            self.ref_dof_vel[stand_mask] = 0.0
            self.ref_lin_vel[stand_mask] = 0.0
            self.ref_ang_vel[stand_mask] = 0.0

        # 启动阶段：平滑混合 default → motion（参考随时间推进，不再锁定 frame_0）
        # blend 控制 default → motion 的混合比，motion_ref_time 让参考随 startup 推进 motion
        if startup_mask.any():
            n_startup = startup_mask.sum().item()
            if n_startup > 0:
                startup_duration = getattr(self.cfg.reset, 'startup_duration_s', 2.75)
                timer = self.transition_timer_buf[startup_mask]

                # blend: 控制 default_dof 与 motion_dof 的混合比例
                blend = (timer / startup_duration).clamp(0.0, 1.0)

                # motion_ref_time: 随 startup 时间推进 motion 帧
                # 到 startup 结束时，参考已推进 motion 的 startup_motion_time_equivalent_s 秒
                motion_equiv = getattr(self.cfg.rewards, 'startup_motion_time_equivalent_s', 2.0)
                elapsed_in_startup = timer.cpu().numpy()
                motion_ref_times = np.clip(elapsed_in_startup + motion_equiv, 0, self.motion_lenth)
                traj_idxs = np.array([0] * n_startup)

                frames_motion = self.motion_loader.get_full_frame_at_time_batch(traj_idxs, motion_ref_times)
                motion_dof = self.motion_loader_class.get_joint_pose_batch(frames_motion)
                motion_vel = self.motion_loader_class.get_joint_vel_batch(frames_motion)
                motion_key = self.motion_loader_class.get_tar_toe_pos_local_batch(frames_motion)
                motion_contact = self.motion_loader_class.get_contact_mask_batch(frames_motion)

                b = blend.unsqueeze(1)
                self.ref_dof_pos[startup_mask] = default_dof * (1 - b) + motion_dof * b
                self.ref_dof_vel[startup_mask] = motion_vel * b
                self.ref_key_pos[startup_mask] = self.ref_key_pos[startup_mask] * (1 - b) + motion_key * b
                self.ref_contact_mask[startup_mask] = motion_contact
                self.ref_lin_vel[startup_mask] = 0.0
                self.ref_ang_vel[startup_mask] = 0.0

    def post_physics_step(self):
        """ 检查终止条件，计算观测值和奖励
            调用self._post_physics_step_callback()进行通用计算 
        """
        # 刷新张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 更新计数器
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.last_episode_length_buf = self.episode_length_buf.clone()

        # 准备数量
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.contacts = self.contact_forces[:, self.feet_indices, 2] > 5.
        self.contacts_filt = torch.logical_or(self.contacts, self.last_contacts).float()

        self._post_physics_step_callback()

        # 更新过渡阶段状态
        self._update_transition_phase()

        # 计算观测值、奖励、重置等
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.privileged_obs_buf[env_ids].clone()
        self.reset_idx(env_ids)
        
        # 领域随机化
        if self.cfg.domain_rand.push_robots:
            self._push_robots()
        
        if self.cfg.domain_rand.disturbance:
            self._disturbance_robots()

        self.compute_observations() # 在某些情况下可能需要模拟步骤来刷新某些观测值（例如身体位置）

        # 更新历史动作和速度
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contacts[:] = self.contacts[:]

        # 调试可视化
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        return env_ids, termination_privileged_obs, None
    
    def check_termination(self):
        """ 检查环境是否需要重置 """
        # 检查接触力终止条件
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 5., dim=1)
        # 超时终止条件（超时无终端奖励）
        time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= time_out_buf
        
        # 重力终止条件
        if self.cfg.termination.terminate_by_gravity:
            reset_terminate_by_gravity = torch.norm(self.projected_gravity[:, 0:2], dim=-1) > self.cfg.termination.scales.termination_gravity
            self.reset_buf |= reset_terminate_by_gravity
            
        # 倒下终止条件
        if self.cfg.termination.terminate_by_fallen:
            self.fallen_buf = (self.root_states[:, 2] - self.terrain_h) < self.cfg.termination.scales.termination_height
            self.reset_buf |= self.fallen_buf
            
        # 运动距离过远终止条件
        if self.cfg.termination.terminate_when_motion_far:
            ref_bodies = self.ref_key_pos.reshape(self.num_envs, -1, 3)
            reset_buf_motion_far = torch.any(torch.norm(ref_bodies - self.key_pos_trunc, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)
            self.reset_buf_terminate_by_motion_far = reset_buf_motion_far
            self.reset_buf |= reset_buf_motion_far
            
        # 运动结束终止条件
        if self.cfg.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.reset_buf_terminate_by_motion_end = current_time > self.motion_lenth
            self.time_out_buf |= self.reset_buf_terminate_by_motion_end
    
    def reset_idx(self, env_ids):
        """ 重置某些环境。
            调用self._reset_dofs(env_ids), self._reset_root_states(env_ids), 和 self._resample_commands(env_ids)
            [可选] 调用self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) 和
            记录episode信息
            重置一些缓冲区

        Args:
            env_ids (list[int]): 必须重置的环境ID列表
        """
        if len(env_ids) == 0:
            return
        
        # 清除观测历史
        if self.cfg.env.frame_stack is not None:
            for i in range(self.obs_history.maxlen):
                self.obs_history[i][env_ids] *= 0
                
        # 更新课程
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.use_reward_penalty_curriculum:
            self._update_reward_penalty_curriculum()
        if self.cfg.termination.terminate_when_motion_far and self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum:
            self._update_terminate_when_motion_far_curriculum()
        
        # 重采样运动时间
        self._resample_motion_times(env_ids)
        # 初始化过渡阶段状态（必须在 _reset_* 之前，因为 reset 函数依赖 transition_phase_buf）
        self._init_transition_state(env_ids)
        # 重置机器人状态
        self._reset_dofs_motion(env_ids, self.motion_start_times[env_ids])
        self._reset_root_states_motion(env_ids, self.motion_start_times[env_ids])

        # 重置缓冲区
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.feet_contact_time[env_ids] = 0.
        self.feet_both_contact_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.episode_length_buf[env_ids] = 0.
        self._update_average_episode_length(env_ids)

        # 更新高度测量
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.terrain_h = self._get_ground_heights()

        # 重置随机化属性
        if self.cfg.domain_rand.randomize_gains:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.p_gain_range[0], self.cfg.domain_rand.p_gain_range[1], (len(env_ids), self.num_actions), device=self.device)
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.d_gain_range[0], self.cfg.domain_rand.d_gain_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), self.num_actions), device=self.device)

        self._refresh_actor_rigid_shape_props(env_ids)

        # 填充额外信息
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
            
        # 记录额外的课程信息
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["min_command_x"] = self.command_ranges["lin_vel_x"][0]
            
        # 发送超时信息给算法
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self.extras["episode"]["end_epis_length"] = self.last_episode_length_buf[env_ids]
        
        # 修复重置重力bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
    
    def _resample_motion_times(self, env_ids):
        """重采样运动时间"""
        if len(env_ids) == 0:
            return
        warmup = getattr(self.cfg.rewards, 'motion_warmup_time_s', 0.5)
        if self.cfg.env.test:
            # 测试模式：从 -warmup ~ 0.0 采样，负时间映射到"稳定站立帧"
            self.motion_start_times[env_ids] = torch_rand_float(-warmup, 0.0, (len(env_ids), 1), device=self.device).squeeze(-1)
        else:
            # 训练模式：~10% 的 env 从 -warmup ~ 0.0 采样（让策略学习"从站立过渡到动作起点"），
            # 其余从动作内随机点采样。这样既能快速学完动作，又不会在 reset 瞬间抽搐。
            n_warmup = max(1, int(0.1 * len(env_ids)))
            warmup_ids = env_ids[:n_warmup]
            self.motion_start_times[warmup_ids] = torch_rand_float(-warmup, 0.0, (n_warmup, 1), device=self.device).squeeze(-1)
            rest_ids = env_ids[n_warmup:]
            if len(rest_ids) > 0:
                self.motion_start_times[rest_ids] = torch_rand_float(0.0, self.motion_lenth - self.dt, (len(rest_ids), 1), device=self.device).squeeze(-1)
    
    def _update_average_episode_length(self, env_ids):
        """更新平均episode长度"""
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.last_episode_length_buf[env_ids], dtype=torch.float)
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)

    def compute_reward(self):
        """ 计算奖励
            调用每个具有非零缩放比例的奖励函数（在self._prepare_reward_function()中处理）
            将每个项添加到episode总和和总奖励中
        """
        self.rew_buf[:] = 0.
        # 遍历所有奖励函数
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # 奖励惩罚课程处理
            if name in self.cfg.rewards.reward_penalty_reward_names:
                if self.cfg.rewards.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale
            # 累加奖励
            if self.cfg.rewards.use_vec_reward:
                self.rew_buf[:,i] += rew
            else:
                self.rew_buf += rew
            self.episode_sums[name] += rew
            
        # 只保留正奖励
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            
        # 添加终止奖励（裁剪后）
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def _update_reward_penalty_curriculum(self):
        """
        根据当前 iteration 的平均 episode 长度更新惩罚课程。

        current_mean_episode_length 由 runner 在每个 iteration 结束时设置，
        反映当前策略的真实表现（而非 EMA 平滑后的历史均值）。
        这样 curriculum 能在单个 episode 内快速响应策略变化。
        """
        if self.current_mean_episode_length < self.cfg.rewards.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= (1 - self.cfg.rewards.reward_penalty_degree)
        elif self.current_mean_episode_length > self.cfg.rewards.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= (1 + self.cfg.rewards.reward_penalty_degree)

        self.reward_penalty_scale = np.clip(self.reward_penalty_scale, self.cfg.rewards.reward_min_penalty_scale, self.cfg.rewards.reward_max_penalty_scale)
    
    def _update_terminate_when_motion_far_curriculum(self):
        """更新运动距离过远终止课程

        使用 current_mean_episode_length（当前 iteration 均值）而非 EMA，
        使 curriculum 能快速响应策略变化。
        """
        assert self.cfg.termination.terminate_when_motion_far and self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum
        if self.current_mean_episode_length < self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif self.current_mean_episode_length > self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        self.terminate_when_motion_far_threshold = np.clip(
            self.terminate_when_motion_far_threshold,
            self.cfg.termination.termination_curriculum.terminate_when_motion_far_threshold_min,
            self.cfg.termination.termination_curriculum.terminate_when_motion_far_threshold_max)

    def get_curriculum_state(self):
        """
        获取当前 curriculum 状态，用于保存到 checkpoint。
        包含 reward_penalty_scale、terminate_when_motion_far_threshold、average_episode_length。
        注意: average_episode_length 可能是 CUDA tensor，需转为 float。
        """
        avg_ep = self.average_episode_length
        if hasattr(avg_ep, 'cpu'):
            avg_ep = float(avg_ep.cpu())
        return {
            "reward_penalty_scale": float(self.reward_penalty_scale),
            "terminate_when_motion_far_threshold": float(self.terminate_when_motion_far_threshold),
            "average_episode_length": float(avg_ep),
        }

    def set_curriculum_state(self, state):
        """
        从 checkpoint 恢复 curriculum 状态。
        确保 resume 后 penalty scale 和 threshold 与 checkpoint 保存时一致，
        避免从头开始 curriculum 导致 policy 初期剧烈振荡。
        """
        self.reward_penalty_scale = float(state["reward_penalty_scale"])
        self.terminate_when_motion_far_threshold = float(state["terminate_when_motion_far_threshold"])
        self.average_episode_length = float(state["average_episode_length"])
        print(f'  [n2_mimic_env] curriculum restored:')
        print(f'    reward_penalty_scale={self.reward_penalty_scale:.6f}')
        print(f'    terminate_when_motion_far_threshold={self.terminate_when_motion_far_threshold:.4f}')
        print(f'    average_episode_length={self.average_episode_length:.2f}')

    def reset_curriculum(self):
        """
        将 curriculum 重置为初始状态。
        用于 resume 时放弃旧 curriculum，让新配置的 curriculum 参数从头开始生效。
        新参数（degree、thresholds）从初始值开始，更能适应 policy 变化。
        """
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = self.cfg.rewards.reward_initial_penalty_scale
        if self.cfg.termination.terminate_when_motion_far and self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum:
            self.terminate_when_motion_far_threshold = self.cfg.termination.termination_curriculum.terminate_when_motion_far_initial_threshold
        self.average_episode_length = float(self.cfg.rewards.num_compute_average_epl)
        self.current_mean_episode_length = 0.
        print(f'  [n2_mimic_env] curriculum RESET to initial values:')
        print(f'    reward_penalty_scale={self.reward_penalty_scale:.6f}')
        print(f'    terminate_when_motion_far_threshold={self.terminate_when_motion_far_threshold:.4f}')
        print(f'    average_episode_length={self.average_episode_length:.2f}')

    def _init_transition_state(self, env_ids):
        """初始化过渡阶段状态：所有 env 从站立稳定阶段开始"""
        self.transition_phase_buf[env_ids] = 0  # 0 = stand_stable
        self.transition_timer_buf[env_ids] = 0.0

    def _update_transition_phase(self):
        """每步更新过渡阶段状态：0=站立稳定 → 1=启动过渡 → 2=正常motion跟踪"""
        if not hasattr(self, 'transition_phase_buf'):
            return
        dt = self.dt  # dt = sim.dt (control loop 中的 actual physics dt)

        stand_mask = (self.transition_phase_buf == 0)
        if stand_mask.any():
            stand_duration = getattr(self.cfg.reset, 'stand_stable_duration_s', 0.75)
            self.transition_timer_buf[stand_mask] += dt
            switch_to_startup = stand_mask & (self.transition_timer_buf >= stand_duration)
            if switch_to_startup.any():
                self.transition_phase_buf[switch_to_startup] = 1
                self.transition_timer_buf[switch_to_startup] = 0.0

        startup_mask = (self.transition_phase_buf == 1)
        if startup_mask.any():
            startup_duration = getattr(self.cfg.reset, 'startup_duration_s', 2.75)
            self.transition_timer_buf[startup_mask] += dt
            switch_to_motion = startup_mask & (self.transition_timer_buf >= startup_duration)
            if switch_to_motion.any():
                self.transition_phase_buf[switch_to_motion] = 2

    def get_rng_state(self):
        """
        获取所有 RNG 状态，用于保存到 checkpoint。
        包含: torch CPU RNG、CUDA RNG（转为 CPU ByteTensor 以便序列化）。
        """
        import random
        cuda_state = torch.cuda.get_rng_state_all()
        # get_rng_state_all() 在多 GPU 时返回 list，单 GPU 时返回 ByteTensor
        # 统一转为 CPU ByteTensor list 便于序列化
        if isinstance(cuda_state, (list, tuple)):
            cuda_state = [s.cpu().to(torch.uint8) for s in cuda_state]
        else:
            cuda_state = [cuda_state.cpu().to(torch.uint8)]
        state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": cuda_state,
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        return state

    def set_rng_state(self, state):
        """
        从 checkpoint 恢复所有 RNG 状态。
        确保 resume 后随机数序列与保存时一致，
        避免因 domain randomization 差异导致初期性能崩溃。

        注意: torch.load(map_location=device) 会把所有 tensor 移到目标 device，
        包括 list 中的 CUDA tensor 和 CPU RNG state。因此需要强制移回 CPU。
        """
        import random
        # CPU RNG：必须先移到 CPU 再使用
        cpu_state = state["torch_cpu"]
        if isinstance(cpu_state, torch.Tensor):
            cpu_state = cpu_state.cpu()
        if not isinstance(cpu_state, torch.ByteTensor):
            cpu_state = torch.ByteTensor(cpu_state.numpy() if hasattr(cpu_state, 'numpy') else list(cpu_state))
        torch.set_rng_state(cpu_state)
        # CUDA RNG：torch.load(map_location) 可能把 list 中的 tensor 移到 CUDA，
        # 需要先移回 CPU 再恢复到对应设备
        cuda_state = state["torch_cuda"]
        if isinstance(cuda_state, (list, tuple)):
            for i, s in enumerate(cuda_state):
                s_cpu = s.cpu()
                torch.cuda.set_rng_state(s_cpu.to(dtype=torch.uint8), device=i)
        else:
            s_cpu = cuda_state.cpu()
            torch.cuda.set_rng_state(s_cpu.to(dtype=torch.uint8))
        np.random.set_state(state["numpy"])
        random.setstate(state["random"])
        print(f'  [n2_mimic_env] RNG state restored')
    
    def compute_observations(self):
        """ 计算观测值 """
        # 获取相位信息
        phase = self._get_phase().unsqueeze(1)
        self.compute_ref_state()

        # 计算正弦和余弦相位
        sin_pos = torch.sin(2 * torch.pi * phase)
        cos_pos = torch.cos(2 * torch.pi * phase)

        # 构建基础观测缓冲区
        obs_buf = torch.cat((   sin_pos,                                        # 正弦相位
                                cos_pos,                                       # 余弦相位
                                self.base_ang_vel  * self.obs_scales.ang_vel,  # 基座角速度
                                self.projected_gravity,                        # 投影重力
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差
                                self.dof_vel * self.obs_scales.dof_vel,        # 关节速度
                                self.actions                                   # 当前动作
                            ),dim=-1)
        
        # 计算关键点位置
        root_pos = self.root_states[:, :3].clone() 
        key_pos = (self.key_pos - root_pos.unsqueeze(1)).view(self.num_envs, -1) 
        ref_key_pos = self.ref_key_pos.clone()
        
        # 构建特权观测缓冲区
        self.privileged_obs_buf = torch.cat((  
                                    sin_pos,                                        # 正弦相位
                                    cos_pos,                                        # 余弦相位
                                    self.base_ang_vel  * self.obs_scales.ang_vel,   # 基座角速度
                                    self.projected_gravity,                         # 投影重力
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差
                                    self.dof_vel * self.obs_scales.dof_vel,         # 关节速度
                                    self.actions,                                   # 当前动作
                                    self.base_lin_vel * self.obs_scales.lin_vel,    # 基座线速度
                                    self.payload * 0.5,                             # 负载信息
                                    self.friction_coeffs,                           # 摩擦系数
                                    self.restitution_coeffs,                        # 恢复系数
                                    self.Kp_factors,                                # Kp因子
                                    self.Kd_factors,                                # Kd因子
                                    self.motor_strength,                            # 电机强度
                                    key_pos,                                        # 关键点位置
                                    ref_key_pos,                                    # 参考关键点位置
                                    self.contacts_filt                              # 接触滤波器
                                ),dim=-1)
        
        # 如果配置了地形高度测量，添加高度信息
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # 根据需要添加噪声
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        
        # 处理帧堆叠
        if self.cfg.env.frame_stack is not None:
            self.obs_history.append(obs_now)
            # 堆叠历史观测
            obs_buf_all = torch.stack([self.obs_history[i]
                                    for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
            self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        else:
            self.obs_buf = obs_now

# ================================================ Sigma ================================================== #
    def _init_adaptive_sigma(self):
        """初始化自适应sigma"""
        # 如果未启用自适应跟踪sigma，将更新函数设置为空函数
        if not self.cfg.rewards.enable_adaptive_tracking_sigma:
            self._update_adaptive_sigma = lambda *args, **kwargs: None
            return
        # 初始化奖励误差指数移动平均
        self._reward_error_ema = dict()
        for key, value in self.cfg.rewards.reward_tracking_sigma.items():
            self._reward_error_ema[key] = value
    
    def _update_adaptive_sigma(self, error:torch.Tensor, term:str):
        """更新自适应sigma"""
        alpha = self.cfg.rewards.tracking_sigma_alpha   # 指数移动平均alpha参数
        scale = self.cfg.rewards.tracking_sigma_scale   # 缩放因子
        adptype = self.cfg.rewards.tracking_sigma_type  # 更新类型
        
        # 更新误差指数移动平均
        self._reward_error_ema[term] = self._reward_error_ema[term] * (1-alpha) + error.mean().item() * alpha
        
        # 根据类型更新sigma
        if adptype == "scale":
            self.cfg.rewards.reward_tracking_sigma[term] = min(self._reward_error_ema[term] * scale, 
                                                                self.cfg.rewards.reward_tracking_sigma[term])
        elif adptype == "mean":
            self.cfg.rewards.reward_tracking_sigma[term] = (min(self._reward_error_ema[term], 
                                                                self.cfg.rewards.reward_tracking_sigma[term]) +
                                                                self._reward_error_ema[term]) / 2
        elif adptype == "origin":
            self.cfg.rewards.reward_tracking_sigma[term] = min(self._reward_error_ema[term], 
                                                                self.cfg.rewards.reward_tracking_sigma[term])
    
    def _reset_dofs_motion(self, env_ids, motion_times):
        """重置关节运动。过渡阶段（phase 0/1）使用 URDF 默认姿态。"""
        if len(env_ids) == 0:
            return

        stand_envs = (self.transition_phase_buf[env_ids] == 0)
        startup_envs = (self.transition_phase_buf[env_ids] == 1)
        motion_envs = (self.transition_phase_buf[env_ids] == 2)

        # 站立稳定阶段：使用 URDF 默认关节
        if stand_envs.any():
            s_ids = env_ids[stand_envs]
            self.dof_pos[s_ids] = self.default_dof_pos
            self.dof_vel[s_ids] = 0.0
            s_int32 = s_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(s_int32), len(s_int32))

        # 启动阶段：平滑混合 default → motion frame 0
        if startup_envs.any():
            u_ids = env_ids[startup_envs]
            startup_duration = getattr(self.cfg.reset, 'startup_duration_s', 2.75)
            blend = (self.transition_timer_buf[u_ids] / startup_duration).clamp(0.0, 1.0).unsqueeze(1)
            # 获取 motion frame 0
            n_startup = startup_envs.sum().item()
            frames_0 = self.motion_loader.get_full_frame_at_time_batch(
                np.array([0] * n_startup), np.zeros(n_startup))
            motion_dof = self.motion_loader_class.get_joint_pose_batch(frames_0)
            blended = self.default_dof_pos * (1 - blend) + motion_dof * blend
            self.dof_pos[u_ids] = blended
            self.dof_vel[u_ids] = 0.0
            u_int32 = u_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(u_int32), len(u_int32))

        # 正常 motion 阶段
        if motion_envs.any():
            m_ids = env_ids[motion_envs]
            m_times = motion_times[motion_envs]
            traj_idxs = np.array([0] * len(m_ids))
            times = m_times.clone().cpu().numpy()
            frames = self.motion_loader.get_full_frame_at_time_batch(traj_idxs, times)
            self.dof_pos[m_ids] = self.motion_loader_class.get_joint_pose_batch(frames)
            self.dof_vel[m_ids] = 0.0
            m_int32 = m_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(m_int32), len(m_int32))

    def _reset_root_states_motion(self, env_ids, motion_times):
        """重置 ROOT 状态位置和速度。过渡阶段使用 URDF 默认站立状态。"""
        if len(env_ids) == 0:
            return

        stand_envs = (self.transition_phase_buf[env_ids] == 0)
        startup_envs = (self.transition_phase_buf[env_ids] == 1)
        motion_envs = (self.transition_phase_buf[env_ids] == 2)
        default_z = self.cfg.init_state.pos[2]
        env_origins_xy = self.env_origins[env_ids, :3] + 0.02

        # 站立稳定阶段：使用 URDF 默认站立状态（默认高度、正面朝上）
        if stand_envs.any():
            s_ids = env_ids[stand_envs]
            origins_xy = self.env_origins[s_ids, :3] + 0.02
            self.root_states[s_ids, 0] = origins_xy[:, 0]
            self.root_states[s_ids, 1] = origins_xy[:, 1]
            self.root_states[s_ids, 2] = default_z
            self.root_states[s_ids, 3:7] = torch.tensor([0., 0., 0., 1.], device=self.device)  # 默认朝向（正面朝上）
            self.root_states[s_ids, 7:13] = 0.0
            s_int32 = s_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_states),
                                                        gymtorch.unwrap_tensor(s_int32), len(s_int32))

        # 启动阶段：平滑混合 default → motion frame 0 的根状态
        if startup_envs.any():
            u_ids = env_ids[startup_envs]
            startup_duration = getattr(self.cfg.reset, 'startup_duration_s', 2.75)
            blend = (self.transition_timer_buf[u_ids] / startup_duration).clamp(0.0, 1.0).unsqueeze(1)
            n_startup = startup_envs.sum().item()
            frames_0 = self.motion_loader.get_full_frame_at_time_batch(
                np.array([0] * n_startup), np.zeros(n_startup))
            motion_root_pos = self.motion_loader_class.get_root_pos_batch(frames_0)
            motion_root_orn = self.motion_loader_class.get_root_rot_batch(frames_0)
            origins_xy = self.env_origins[u_ids, :3] + 0.02
            # 混合位置：default (x=0,y=0) → motion frame 0
            default_root = torch.zeros(n_startup, 3, device=self.device)
            blended_root_pos = default_root * (1 - blend) + motion_root_pos * blend
            blended_root_pos[:, 2] = default_z  # z 始终为默认高度
            blended_root_pos[:, 0] += origins_xy[:, 0]
            blended_root_pos[:, 1] += origins_xy[:, 1]
            # 混合朝向
            blended_root_orn = self._quat_slerp(
                torch.tensor([0., 0., 0., 1.], device=self.device).expand(n_startup, -1),
                motion_root_orn,
                blend.squeeze(1)
            )
            self.root_states[u_ids, :3] = blended_root_pos
            self.root_states[u_ids, 3:7] = blended_root_orn
            self.root_states[u_ids, 7:13] = 0.0
            u_int32 = u_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_states),
                                                        gymtorch.unwrap_tensor(u_int32), len(u_int32))

        # 正常 motion 阶段
        if motion_envs.any():
            m_ids = env_ids[motion_envs]
            m_times = motion_times[motion_envs]
            traj_idxs = np.array([0] * len(m_ids))
            times = m_times.clone().cpu().numpy()
            frames = self.motion_loader.get_full_frame_at_time_batch(traj_idxs, times)
            root_pos = self.motion_loader_class.get_root_pos_batch(frames)
            root_pos[:, 2] = default_z
            root_pos[:, :3] = root_pos[:, :3] + self.env_origins[m_ids, :3] + 0.02
            self.root_states[m_ids, :3] = root_pos
            self.root_states[m_ids, 3:7] = self.motion_loader_class.get_root_rot_batch(frames)
            self.root_states[m_ids, 7:13] = 0.0
            m_int32 = m_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_states),
                                                        gymtorch.unwrap_tensor(m_int32), len(m_int32))

    def _quat_slerp(self, q1, q2, t):
        """球面线性插值（SLERP）"""
        dot = (q1 * q2).sum(dim=-1)
        dot_clamped = dot.clamp(-1.0, 1.0)
        omega = torch.acos(dot_clamped)
        sin_omega = torch.sin(omega)
        sin_omega[sin_omega < 1e-5] = 1.0
        w1 = torch.sin((1 - t) * omega) / sin_omega
        w2 = torch.sin(t * omega) / sin_omega
        w1, w2 = w1.unsqueeze(-1), w2.unsqueeze(-1)
        return w1 * q1 + w2 * q2
        
# ================================================ Rewards ================================================== #
    def _reward_tracking_body_pos(self):
        """身体位置跟踪奖励"""
        ref_key_pos = self.ref_key_pos.reshape(self.num_envs, -1, 3)
        num_ref = ref_key_pos.shape[1]

        upper = torch.clamp(self.upper_body_indices, 0, num_ref - 1)
        lower = torch.clamp(self.lower_body_indices, 0, num_ref - 1)

        upper_body_diff = ref_key_pos[:, upper, :] - self.key_pos_trunc[:, upper, :]
        lower_body_diff = ref_key_pos[:, lower, :] - self.key_pos_trunc[:, lower, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.cfg.rewards.reward_tracking_sigma["tracking_upper_body_pos"])
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.cfg.rewards.reward_tracking_sigma["tracking_lower_body_pos"])
        rew = r_body_pos_lower + r_body_pos_upper

        self._update_adaptive_sigma(diff_body_pos_dist_upper, 'tracking_upper_body_pos')
        self._update_adaptive_sigma(diff_body_pos_dist_lower, 'tracking_lower_body_pos')
        return rew

    def _reward_tracking_feet_pos(self):
        """足部位置跟踪奖励"""
        ref_key_pos = self.ref_key_pos.reshape(self.num_envs, -1, 3)
        num_ref = ref_key_pos.shape[1]

        feet = torch.clamp(self.feet_indices, 0, num_ref - 1)

        feet_diff = ref_key_pos[:, feet, :] - self.key_pos_trunc[:, feet, :]
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        rew = torch.exp(-feet_dist / self.cfg.rewards.reward_tracking_sigma["tracking_feet_pos"])

        self._update_adaptive_sigma(feet_dist, 'tracking_feet_pos')
        return rew
    
    def _reward_tracking_body_vel(self):
        """身体速度跟踪奖励"""
        body_vel = self.base_lin_vel.clone()
        body_vel_target = self.ref_lin_vel.clone()
        diff = ((body_vel - body_vel_target) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_body_vel"])
        self._update_adaptive_sigma(diff, 'tracking_body_vel')
        return rew
    
    def _reward_tracking_body_ang_vel(self):
        """身体角速度跟踪奖励"""
        body_ang_vel = self.base_ang_vel.clone()
        body_ang_vel_target = self.ref_ang_vel.clone()
        diff = ((body_ang_vel - body_ang_vel_target) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_body_ang_vel"]) 
        self._update_adaptive_sigma(diff, 'tracking_body_ang_vel')
        return rew
    
    def _reward_tracking_joint_pos(self):
        """关节位置跟踪奖励"""
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        # 将踝关节位置设为0
        joint_pos[:, self.ankle_dof_idxs] = 0.0
        pos_target[:, self.ankle_dof_idxs] = 0.0
        diff = ((joint_pos - pos_target) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_joint_pos"])
        self._update_adaptive_sigma(diff, 'tracking_joint_pos')
        return rew
    
    def _reward_tracking_joint_vel(self):
        """关节速度跟踪奖励"""
        joint_vel = self.dof_vel.clone()
        vel_target = self.ref_dof_vel.clone()
        # 将踝关节速度设为0
        joint_vel[:, self.ankle_dof_idxs] = 0.0
        vel_target[:, self.ankle_dof_idxs] = 0.0
        diff = ((joint_vel - vel_target) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_joint_vel"])
        self._update_adaptive_sigma(diff, 'tracking_joint_vel')
        return rew
    
    def _reward_tracking_max_joint_pos(self):
        """最大关节位置跟踪奖励"""
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        # 将踝关节位置设为0
        joint_pos[:, self.ankle_dof_idxs] = 0.0
        pos_target[:, self.ankle_dof_idxs] = 0.0
        # 计算最大关节位置差异
        max_diff_joint_pos = ((joint_pos - pos_target).abs()).max(dim=-1)[0]
        r_max_joint_pos = torch.exp(-max_diff_joint_pos / self.cfg.rewards.reward_tracking_sigma["tracking_max_joint_pos"])
        
        # 更新自适应sigma
        self._update_adaptive_sigma(max_diff_joint_pos, 'tracking_max_joint_pos')
        return r_max_joint_pos

    # ---- Boxing3: 手臂关节跟踪（优先级最高）----
    def _reward_tracking_arm_joint_pos(self):
        """手臂关节位置跟踪奖励（仅手臂 DOFs：8 个关节）

        拳击动作中手臂跟踪最关键：shoulder pitch/roll/yaw + elbow 决定出拳动作。
        """
        arm_idx = self.arm_dof_idxs
        cur = self.dof_pos[:, arm_idx]
        ref = self.ref_dof_pos[:, arm_idx]
        diff = ((cur - ref) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_arm_joint_pos"])
        self._update_adaptive_sigma(diff, 'tracking_arm_joint_pos')
        return rew

    def _reward_tracking_arm_joint_vel(self):
        """手臂关节速度跟踪奖励（仅手臂 DOFs）

        保证出拳动作流畅、平滑。
        """
        arm_idx = self.arm_dof_idxs
        cur = self.dof_vel[:, arm_idx]
        ref = self.ref_dof_vel[:, arm_idx]
        diff = ((cur - ref) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_arm_joint_vel"])
        self._update_adaptive_sigma(diff, 'tracking_arm_joint_vel')
        return rew

    def _reward_tracking_arm_max_joint_pos(self):
        """手臂最大关节位置偏差奖励（仅手臂 DOFs）

        惩罚最偏差最大的单个手臂关节。
        """
        arm_idx = self.arm_dof_idxs
        cur = self.dof_pos[:, arm_idx]
        ref = self.ref_dof_pos[:, arm_idx]
        max_diff = ((cur - ref).abs()).max(dim=-1)[0]
        rew = torch.exp(-max_diff / self.cfg.rewards.reward_tracking_sigma["tracking_arm_max_joint_pos"])
        self._update_adaptive_sigma(max_diff, 'tracking_arm_max_joint_pos')
        return rew

    # ---- Boxing3: 腿部关节跟踪（平衡支撑）----
    def _reward_tracking_leg_joint_pos(self):
        """腿部关节位置跟踪奖励（仅腿部 DOFs：10 个关节）

        拳击动作中腿部支撑决定稳定性：hip yaw/roll/pitch + knee + ankle。
        """
        leg_idx = self.leg_dof_idxs
        cur = self.dof_pos[:, leg_idx]
        ref = self.ref_dof_pos[:, leg_idx]
        diff = ((cur - ref) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_leg_joint_pos"])
        self._update_adaptive_sigma(diff, 'tracking_leg_joint_pos')
        return rew

    def _reward_tracking_leg_joint_vel(self):
        """腿部关节速度跟踪奖励（仅腿部 DOFs）

        保证站立和移动时的平衡感。
        """
        leg_idx = self.leg_dof_idxs
        cur = self.dof_vel[:, leg_idx]
        ref = self.ref_dof_vel[:, leg_idx]
        diff = ((cur - ref) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_leg_joint_vel"])
        self._update_adaptive_sigma(diff, 'tracking_leg_joint_vel')
        return rew

    def _reward_tracking_contact_mask(self):
        """接触掩码跟踪奖励"""
        cur_contact_mask = self.contacts_filt
        ref_contact_mask = self.ref_contact_mask
        
        # 计算接触掩码误差
        error_contact_mask = (cur_contact_mask - ref_contact_mask).abs()

        rew = 1 - error_contact_mask.mean(dim=-1)
        return rew

    def _reward_feet_air_time(self):
        """足部悬空时间奖励"""
        # 奖励长步幅
        # 需要过滤接触，因为PhysX在网格上的接触报告不可靠
        contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # 仅在首次接触地面时给予奖励
        rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_motion_incentive(self):
        """运动激励奖励：鼓励机器人主动运动，打破「原地站立」的局部最优。

        在 stand/startup 阶段 ref_vel=0 时，tracking 奖励不起作用。
        该奖励对非零关节速度给予正奖励，推动机器人主动跟随 motion。
        """
        # 使用手臂关节速度为主（拳击核心在手臂动作）
        arm_vel = self.dof_vel[:, self.arm_dof_idxs]
        arm_vel_mag = torch.norm(arm_vel, dim=1)
        rew = torch.clamp(arm_vel_mag - 0.2, 0.0, 10.0) * 0.5
        return rew

    def _reward_base_tilt_asymmetric(self):
        """非对称身体倾斜惩罚：允许少量前倾，严厉惩罚后倾。

        projected_gravity[1] = pitch：
          > 0 → 身体前倾（拳击出拳姿态，正常）
          < 0 → 身体后倾（倒退回退，问题行为）
        非对称设计：前倾容忍 0.3 rad，后倾容忍 0.05 rad。
        """
        pitch = self.projected_gravity[:, 1]

        # 前倾惩罚（轻微）：pitch > 0.3 rad 时开始惩罚
        fwd_pen = torch.clamp(pitch - 0.3, 0.0, 10.0)

        # 后倾惩罚（严厉）：pitch < -0.05 rad 时开始惩罚，平方放大
        bwd_pen = torch.clamp(-pitch - 0.05, 0.0, 10.0) ** 2

        # 系数：前倾容忍度大，后倾几乎零容忍
        rew = -0.5 * fwd_pen - 5.0 * bwd_pen
        return rew

    def _reward_yaw_stability(self):
        """Yaw 轴稳定性惩罚：允许小幅扭腰，严格限制大幅转向。

        只惩罚 yaw 角速度超过阈值（0.2 rad/s ≈ 11°/s）的情况。
        拳击 motion 的 yaw 角速度几乎为 0，因此该奖励约束任意显著 yaw 旋转。
        """
        yaw_vel = self.base_ang_vel[:, 2]
        ref_yaw_vel = self.ref_ang_vel[:, 2]
        error = torch.clamp((yaw_vel - ref_yaw_vel).abs() - 0.2, 0.0, 10.0) ** 2
        return -1.0 * error