# N2 Walking_3 训练逻辑详解

> 基于 `noetix_n2_gym` IsaacGym RL 框架，覆盖从 mocap 动作数据导入到 PPO 策略网络的完整数据流与参数配置。

---

## 1. 系统架构总览

```
mocap JSON (Walking_3.json)
        ↓
MotionLoaderNingTracking          ← 动作数据解析 + 采样
        ↓
N2MimicEnv                      ← IsaacGym 仿真环境 + 奖励计算
        ↓
OnPolicyRunner                   ← PPO 训练编排
        ↓
ActorCritic (PPO)               ← 策略网络 (Actor) + 价值网络 (Critic)
        ↓
训练结果 (model_*.pt)
```

---

## 2. 动作数据导入

### 2.1 原始数据格式

**文件**: `datasets/mocap_motions/ning/walking/Walking_3.json`

| 属性 | 值 |
|---|---|
| 帧数 | 588 帧 |
| 帧率 | 29.96 fps |
| 总时长 | **19.59 秒** |
| 循环模式 | `Wrap` |
| MotionWeight | 0.5 |
| 每帧维度 | **111 维** |

### 2.2 帧结构（111 → 62 维有效数据）

mocap 原始 111 维经过 `MotionLoaderNingTracking` 处理，取前 **62 维**：

| 段 | 索引范围 | 维度 | 内容 |
|---|---|---|---|
| `root_pos` | `[0:3]` | 3 | 根部位置 (x, y, z) |
| `root_rot` | `[3:7]` | 4 | 根部四元数 (归一化后) |
| `joint_pose` | `[7:25]` | **18** | 18 个关节角度 |
| `tar_toe_pos_local` | `[25:85]` | 60 | 20 个关键点局部坐标 (20×3) |
| `lin_vel` | `[37:40]` | 3 | 根部线速度 |
| `ang_vel` | `[40:43]` | 3 | 根部角速度 |
| `joint_vel` | `[43:61]` | 18 | 关节速度 |
| `base_height` | `[61:62]` | 1 | 根部高度 |
| `contact_mask` | `[62:64]` | 2 | 双脚接触标志 |

> `tar_toe_pos_local` 在 `compute_ref_state()` 中通过 `key_pos_full[:, :num_ref_bodies, :]` 截取为 20×3，仅取前 20 个关键点。

### 2.3 Walking_3 起点姿态（与默认站立对比）

Walking_3 的 **frame 0** 与 `default_joint_angles` 的偏差远小于 Boxing_3：

| 部位 | 默认站立 (rad) | Walking_3 t=0 (rad) | 偏差 |
|---|---|---|---|
| L_leg_hip_pitch | -0.15 | +0.14 | 0.28 |
| L_leg_knee | 0.32 | +0.12 | 0.20 |
| L_leg_ankle | -0.17 | -0.24 | 0.07 |
| root_z | 0.75 | **0.657** | **-9.3 cm** |

Walking_3 起点 root_z=0.657 仍低于默认 0.75（略低于 Boxing_3 的 0.641），但整体偏差更小。

### 2.4 MotionLoaderNingTracking 核心机制

```python
# humanoid/amp_utils/motion_loader.py

class MotionLoaderNingTracking:
    # 预加载：采样 1000000 个随机 (traj_idx, time) 对，构建观测序列
    def __init__(..., num_preload_transitions=10, reference_observation_horizon=2):
        # ...
        for i in range(reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames * i
            )

    # 核心采样：给定 env_ids 和 motion_times，返回对应的完整帧
    def get_full_frame_at_time_batch(self, traj_idxs, times):
        # times < 0 → warmup 帧（默认站立姿态，vel=0，contact=1,1）
        # times ≥ 0 → slerp 插值到 motion 轨迹上对应时间点
```

训练时 `_resample_motion_times()` 采样策略：
```python
# humanoid/envs/n2/n2_mimic_env.py
def _resample_motion_times(self, env_ids):
    # 10% 从 [-warmup, 0) 采样（站立热身 → 动作起点过渡）
    # 90% 从 [0, motion_length - dt) 随机起始
    warmup_ids = env_ids[:int(0.1 * len(env_ids))]
    self.motion_start_times[warmup_ids] = torch_rand_float(-warmup, 0.0, ...)
    rest_ids = env_ids[int(0.1 * len(env_ids)):]
    self.motion_start_times[rest_ids] = torch_rand_float(0.0, motion_lenth - dt, ...)
```

---

## 3. 环境与状态空间

### 3.1 N2MimicEnv 继承链

```
LeggedRobot         (IsaacGym 基础仿真)
    ↓
N2Env               (N2 机器人专用：18-DOF、foot state、PD 控制)
    ↓
N2MimicEnv          (动作模仿：motion_loader、参考状态、跟踪奖励)
```

### 3.2 关节配置

**18 自由度** (`num_actions = 18`)：

| 索引 | 关节名称 | 刚度 (N·m/rad) | 阻尼 (N·m·s/rad) |
|---|---|---|---|
| 0 | L_arm_shoulder_pitch | 30 | 1.0 |
| 1 | L_arm_shoulder_roll | 30 | 1.0 |
| 2 | L_arm_shoulder_yaw | 30 | 1.0 |
| 3 | L_arm_elbow | 30 | 1.0 |
| 4 | L_leg_hip_yaw | 80 | 5.0 |
| 5 | L_leg_hip_roll | 80 | 5.0 |
| 6 | L_leg_hip_pitch | 120 | 5.0 |
| 7 | L_leg_knee | 120 | 5.0 |
| 8 | L_leg_ankle | 20 | 2.0 |
| 9–12 | R_arm_* | (同上) | (同上) |
| 13–17 | R_leg_* | (同上) | (同上) |

### 3.3 观测空间 (310 维)

```
每帧 62 维，frame_stack = 5 → 总观测 = 62 × 5 = 310 维
```

| 字段 | 索引范围 | 维度 | 归一化 |
|---|---|---|---|
| 命令 (lin_vel_x, lin_vel_y, ang_vel_yaw) | `[0:3]` | 3 | 无 |
| 根部角速度 (body_ang_vel) | `[3:6]` | 3 | ×1.0 |
| 重力方向 (projected_gravity) | `[6:9]` | 3 | 无 |
| 关节位置偏差 (dof_pos - default) | `[9:27]` | 18 | ×1.0 |
| 关节速度 (dof_vel) | `[27:45]` | 18 | ×0.05 |
| 上一时刻动作 (last_actions) | `[45:63]` | 18 | 无 |

> 噪声添加：dof_pos ±5%、dof_vel ±50%、ang_vel ±20%、gravity ±5%、quat ±5%

### 3.4 仿真步长

```
dt = 0.002s (IsaacGym 仿真步长)
decimation = 10 (每策略 step 仿真 10 步)
策略 dt = 0.002 × 10 = 0.02s (50 Hz)
episode_length_s = 20 → 每回合最多 1000 步
```

---

## 4. PPO 神经网络架构

### 4.1 ActorCritic 类 (actor_critic.py)

```python
class ActorCritic(nn.Module):
    is_recurrent = False  # 前馈网络，无 RNN

    def __init__(self,
        num_actor_obs=310,      # 观测维度 (输入)
        num_critic_obs=310,     # Critic 输入（Walking_3 用 rl 模式 → 同观测）
        num_actions=18,         # 动作维度 (输出)
        actor_hidden_dims=[1024, 256, 128],
        critic_hidden_dims=[768, 256, 128],
        activation="elu",
        init_noise_std=0.5,
    ):
```

### 4.2 Actor 网络（策略网络）

**输入**: 310 维观测向量 → **输出**: 18 维动作均值

```
Observations (310)
        │
        ▼
  Linear(310 → 1024)
        │
        ▼
      ELU()
        │
        ▼
  Linear(1024 → 256)
        │
        ▼
      ELU()
        │
        ▼
   Linear(256 → 128)
        │
        ▼
      ELU()
        │
        ▼
  Linear(128 → 18)   ← 动作均值
        │
        ▼
  DiagGaussian Distribution (用 std 参数构造)
```

**动作分布**: `π(a|s) = N(μ(s), σ²I)`
- 均值: `actor(obs)` 输出 (18,)
- 标准差: `std = 0.5` (可学习标量参数，`nn.Parameter(0.5 * ones(18))`)

### 4.3 Critic 网络（价值网络）

**输入**: 310 维观测向量 → **输出**: 1 维状态价值标量

```
Observations (310)
        │
        ▼
  Linear(310 → 768)
        │
        ▼
      ELU()
        │
        ▼
   Linear(768 → 256)
        │
        ▼
      ELU()
        │
        ▼
   Linear(256 → 128)
        │
        ▼
      ELU()
        │
        ▼
    Linear(128 → 1)   ← 状态价值
        │
        ▼
     Value (scalar)
```

### 4.4 参数规模

| 网络 | 参数量 | 架构 |
|---|---|---|
| Actor | ~354K | 310→1024→256→128→18 |
| Critic | ~406K | 310→768→256→128→1 |
| std 参数 | 18 | 标量可学习 |
| **总计** | **~760K** | — |

---

## 5. PPO 算法核心参数

### 5.1 完整超参数 (N2Walking3CfgPPO)

| 参数 | 值 | 说明 |
|---|---|---|
| `gamma` | 0.99 | 折扣因子 |
| `lam` (GAE) | 0.95 | 广义优势估计参数 |
| `clip_param` | 0.2 | PPO 裁剪区间 ε |
| `num_learning_epochs` | **3** | 每次迭代对同一批数据做 3 轮梯度更新 |
| `num_mini_batches` | **4** | 每个 epoch 分 4 个 mini-batch 更新 |
| `entropy_coef` | 0.005 | 熵正则系数（鼓励探索）|
| `learning_rate` | **3e-4** | 保守学习率 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `desired_kl` | **0.005** | 自适应学习率目标 KL 散度 |
| `use_clipped_value_loss` | True | 价值损失裁剪 |
| `init_noise_std` | **0.5** | 初始动作噪声（保守）|
| `num_steps_per_env` | 24 | 每个 env 收集 24 步后才更新 |

### 5.2 GAE 优势估计

```python
# ppo.py
self.gamma = 0.99      # 折扣
self.lam = 0.95        # GAE lambda

# Rollout: 收集 (num_envs × num_steps_per_env) 个 transition
# Storage 中计算 GAE(lambda):
#   A_t = δ_t + γ·λ·A_{t+1}
#   δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
```

### 5.3 PPO 损失函数

```python
# ppo.py - update()
surrogate_loss = min(
    ratio * advantages,
    clip(ratio, 1-ε, 1+ε) * advantages
)
value_loss = mse(value, returns)
entropy_loss = -entropy_coef * entropy

total_loss = -surrogate_loss + value_loss_coef * value_loss + entropy_loss
```

---

## 6. 奖励函数设计

### 6.1 Walking_3 奖励权重 (N2Walking3Cfg)

```python
class rewards:
    # ===== 跟踪奖励 (正向) =====
    tracking_body_pos        = 1.0    # 20个关键点位置跟踪（body frame）
    tracking_body_vel        = 0.5    # 根部线速度跟踪
    tracking_body_ang_vel    = 0.5    # 根部角速度跟踪
    tracking_feet_pos        = 1.5    # 脚部位置跟踪（权重最高）
    tracking_joint_pos       = 1.0    # 18关节角度跟踪
    tracking_joint_vel       = 1.0    # 关节速度跟踪
    tracking_max_joint_pos   = 1.0    # 最大单关节偏差惩罚
    tracking_contact_mask    = 0.5    # 双脚接触掩码匹配

    # ===== 步态奖励 =====
    feet_air_time            = 1.5    # 腾空时间奖励

    # ===== 惩罚项 (负向) =====
    contact_no_vel           = -5.0   # 有接触但无速度（粘滞检测）
    feet_contact_forces      = -0.01  # 接触力惩罚
    action_rate             = -0.1   # 动作变化率惩罚（平滑）
    action_smoothness       = -0.05  # 动作二阶导惩罚
    dof_acc                 = -2.5e-7  # 加速度惩罚（能耗）
    energy_cost             = -0.001 # 关节力矩惩罚
    collision               = -10.0  # 碰撞惩罚（base/knee/hip/hand/arm）
    dof_pos_limits          = -5.0   # 关节限位惩罚
    dof_vel_limits          = -5.0   # 关节速度限位惩罚
```

### 6.2 自适应 Sigma 机制

```python
# n2_mimic_env.py
enable_adaptive_tracking_sigma = True
tracking_sigma_alpha = 0.001
tracking_sigma_type = "origin"

reward_tracking_sigma = {
    "tracking_upper_body_pos": 0.015,   # 很严格 → 小误差才有高奖励
    "tracking_lower_body_pos": 0.015,
    "tracking_body_vel": 1.0,
    "tracking_body_ang_vel": 15.0,
    "tracking_feet_pos": 0.01,          # 最严格
    "tracking_joint_pos": 0.3,
    "tracking_joint_vel": 30.0,
    "tracking_max_joint_pos": 1.0,
}
# 自适应 sigma: 跟踪误差增大时自动放宽 sigma，降低时收紧
# sigma_t = min(sigma_initial, α * mean_error + (1-α) * sigma_prev)
```

### 6.3 奖励 Penalty Curriculum

```python
reward_penalty_curriculum = True
reward_initial_penalty_scale = 0.1   # 初始惩罚打 1 折 → 让策略先学动作
reward_penalty_level_down_threshold = 100   # ep_len < 100 → 减轻惩罚
reward_penalty_level_up_threshold = 600     # ep_len > 600 → 恢复正常惩罚
reward_penalty_degree = 3e-3              # 每次更新 ±0.3%

reward_penalty_reward_names = [
    "contact_no_vel", "feet_contact_forces",
    "dof_acc", "energy_cost", "action_smoothness",
    "action_rate", "dof_pos_limits", "dof_vel_limits"
]
```

### 6.4 终止条件

```python
terminate_when_motion_far = True   # 关键点偏离过大则终止
terminate_when_motion_far_threshold = 2.5  # 帧距离阈值

# Curriculum:
terminate_when_motion_far_initial_threshold = 1.8  # 开始时宽松
terminate_when_motion_far_threshold_min = 0.8      # 最严格
terminate_when_motion_far_threshold_max = 2.5      # 最宽松
terminate_when_motion_far_curriculum_degree = 3e-3
```

---

## 7. 域随机化 (Domain Randomization)

```python
class domain_rand:
    # 电机特性
    p_gain_range      = [0.8, 1.2]    # PD 刚度 ±20%
    d_gain_range      = [0.8, 1.2]
    motor_strength_range = [0.8, 1.2]  # 驱动力 ±20%

    # 物理参数
    com_displacement_range = [-0.05, 0.05]  # 质心偏移 ±5cm
    friction_range = [0.1, 2.0]             # 摩擦系数 [0.1, 2.0]
    restitution_range = [0.0, 1.0]           # 恢复系数 [0, 1]
    added_mass_range = [-5.0, 5.0]          # 质量 ±5kg

    # 外部扰动
    push_force_range = [50, 300] N          # 随机推力
    push_torque_range = [25, 100] N·m       # 随机扭矩
    disturbance_probabilities = 0.002         # 每步扰动概率 0.2%
    disturbance_interval = [10, 25] 步       # 扰动间隔
```

---

## 8. 训练结果

### 8.1 可用 Run

| Run | 起始 Iter | 状态 |
|---|---|---|
| `0606_00-31-34_` | 0 | Base run |
| `0606_09-03-26_` | — | Continuation |
| `0609_21-37-34_` | — | Continuation |
| `0610_20-31-42_` | 0 | Fresh run |

### 8.2 关键指标（0610_20-31-42_ 末帧 Sigma）

```
sigma.tracking_upper_body_pos = 0.015
sigma.tracking_lower_body_pos = 0.015
sigma.tracking_body_vel       = 0.999
sigma.tracking_body_ang_vel   = 14.99
sigma.tracking_feet_pos       = 0.010
sigma.tracking_joint_pos      = 0.300
sigma.tracking_joint_vel      = 30.0
sigma.tracking_max_joint_pos  = 0.999
```

### 8.3 训练命令

```bash
# 从头训练
python humanoid/scripts/train.py \
  --task n2_walking3 \
  --headless \
  --num_envs 1024 \
  --max_iterations 50000

# 续训
python humanoid/scripts/train.py \
  --task n2_walking3 \
  --resume \
  --load_run 0606_00-31-34_ \
  --checkpoint 24700 \
  --headless \
  --num_envs 1024 \
  --max_iterations 50000

# Play 可视化
python humanoid/scripts/play.py \
  --task n2_walking3 \
  --load_run 0606_00-31-34_ \
  --checkpoint 24700 \
  --num_envs 1 \
  --num_steps 2000
```

---

## 9. 核心文件索引

| 文件 | 职责 |
|---|---|
| `envs/n2/n2_walking3_config.py` | Walking_3 专用超参数 |
| `envs/n2/n2_mimic_config.py` | 动作模仿基类配置（18-DOF, N2MimicEnv）|
| `envs/n2/n2_mimic_env.py` | 环境实现：参考状态计算、奖励、终止条件、curriculum |
| `envs/n2/n2_env.py` | N2 机器人基类：foot state、接触力、PD 控制 |
| `envs/base/legged_robot.py` | IsaacGym 底层仿真框架 |
| `amp_utils/motion_loader.py` | `MotionLoaderNingTracking` 动作数据加载与采样 |
| `algo/ppo/actor_critic.py` | ActorCritic 网络（[1024,256,128] / [768,256,128]）|
| `algo/ppo/ppo.py` | PPO 算法（GAE、裁剪损失、Adam 优化）|
| `algo/ppo/on_policy_runner.py` | 训练循环编排（rollout → compute_returns → update）|
| `algo/ppo/rollout_storage.py` | Transition 存储与 mini-batch 生成 |
| `scripts/train.py` | 训练入口 |
| `scripts/play.py` | Play 可视化入口 |
