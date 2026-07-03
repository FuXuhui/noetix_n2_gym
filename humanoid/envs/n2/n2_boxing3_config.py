from .n2_mimic_config import N2MimicCfg, N2MimicCfgPPO


class N2Boxing3Cfg(N2MimicCfg):
    """N2 Boxing_3 motion imitation config with conservative PPO.

    拳击动作特点：
    - 13s 单动作（392 帧 @ 30fps）
    - 手臂动作幅度大：shoulder pitch/roll/yaw + elbow 活动范围 1-2 rad
    - 腿部支撑重要：出拳时需保持平衡，hip/knee/ankle 活动范围 1-2.3 rad
    - 核心需求：手臂关节精准跟踪 > 腿部关节精准跟踪 > 整体平衡
    """

    class env(N2MimicCfg.env):
        episode_length_s = 14  # ~13s motion + 1s buffer

    # ------------------------------------------------------------------ #
    # termination — 修复 curriculum（与 walking3 一致，但适配13s短动作）
    # ------------------------------------------------------------------ #
    class termination(N2MimicCfg.termination):
        terminate_when_motion_end = False  # 单动作连续循环，不在运动结束时终止

        class scales(N2MimicCfg.termination.scales):
            termination_motion_far_threshold = 10.0   # 与 curriculum max 对齐

        class termination_curriculum(N2MimicCfg.termination.termination_curriculum):
            # ⚠️ boxing3 关闭自动收紧 ——
            # 原 curriculum 会在 ep_len > 200 时一直收紧到 1.5，
            # 导致策略被迫在未稳定时反复被 kill，无法完成完整 14s 出拳动作。
            terminate_when_motion_far_curriculum = False
            terminate_when_motion_far_initial_threshold = 5.0   # 固定宽松阈值（与 curriculum max 对齐）
            terminate_when_motion_far_threshold_min = 5.0
            terminate_when_motion_far_threshold_max = 5.0
            terminate_when_motion_far_curriculum_level_down_threshold = 30
            terminate_when_motion_far_curriculum_level_up_threshold = 200
            terminate_when_motion_far_curriculum_degree = 0.0   # 不再变化

    # ------------------------------------------------------------------ #
    # rewards — 手臂关节优先级 + 平衡保护 + 运动激励
    # ------------------------------------------------------------------ #
    class rewards(N2MimicCfg.rewards):
        motion_warmup_time_s = 0.5

        # 过渡阶段配置
        class transition:
            transition_duration_s = 1.3
            stand_stable_duration_s = 0.3
            startup_duration_s = 1.0

        # 重构：startup 阶段参考随时间推进 motion（不再锁定 frame_0）
        # startup_motion_time_equivalent_s = 1.5 表示：
        #   startup 开始时参考等价于 motion 第 1.5 秒，结束时等价于 motion 第 2.5 秒
        #   让 startup 后期腿部已经开始有跨步动作，给 policy 接触腿部跟踪的早期经验
        startup_motion_time_equivalent_s = 1.5

        reward_penalty_curriculum = True
        reward_initial_penalty_scale = 0.1
        reward_penalty_level_down_threshold = 10
        reward_penalty_level_up_threshold = 500   # 原 200 → 500：让 policy 稳定后再加重负奖励
        reward_penalty_degree = 0.01

        class scales(N2MimicCfg.rewards.scales):
            # ---- 身体位置跟踪 ----
            tracking_body_pos = 2.0
            tracking_body_vel = 1.5
            tracking_body_ang_vel = 1.0
            tracking_feet_pos = 2.0

            # ---- 关节跟踪（arm/leg 分开，整体降低避免重复）----
            tracking_joint_pos = 0.1
            tracking_joint_vel = 0.1
            tracking_max_joint_pos = 0.1       # 原 0.3 → 0.1（max 奖励对极端单关节过敏感）
            tracking_contact_mask = 0.5

            # ---- 拳击专用：手臂关节高精度跟踪（权重大幅提升）----
            # shoulder_pitch/roll/yaw + elbow：拳击动作的核心
            tracking_arm_joint_pos = 5.0     # 降低：避免压制腿部学习
            tracking_arm_joint_vel = 3.0     # 降低：避免压制腿部学习
            tracking_arm_max_joint_pos = 3.0  # 降低：避免压制腿部学习

            # ---- 拳击专用：腿部关节跟踪（与手臂同等重要——大步跨步+退回的步态必须跟）----
            # 权重提升到与手臂同量级，确保步态不被手臂动作压制
            tracking_leg_joint_pos = 5.0      # 提升：腿部步态跟踪（跨步/退回）
            tracking_leg_joint_vel = 3.0      # 提升：腿部速度平滑性

            # ---- 步态/平衡 ----
            feet_air_time = 0.5

            # ---- 运动激励已禁用 ----
            # 重要：boxing3 的目标是"按节律出拳"，不是"主动乱画弧线"。
            # 原 `motion_incentive` 鼓励非零关节速度，会让 policy 用手舞足蹈刷分。
            # 删除后让 tracking 类奖励主导，避免"原地抖手"局部最优。
            motion_incentive = 0.0

            # ---- 身体倾斜：允许前倾，严惩后倾 ----
            base_tilt_asymmetric = 1.0        # 非对称倾斜惩罚（函数内已有惩罚系数）

            # ---- Yaw 轴稳定性：允许小幅扭腰，严限大幅转向 ----
            yaw_stability = 1.0                # yaw 角速度专项惩罚

            # ---- 累积 yaw 漂移惩罚：boxing3 屏蔽 ----
            # 原惩罚在 ep_len > 200 后累积 > 0.26 rad 即严惩，
            # 但 `tracking_base_yaw` + `yaw_stability` 已足，重复惩罚让 yaw 完全无奖励信号。
            yaw_drift_penalty = 0.0            # 屏蔽（保留函数定义以便 resume 兼容）

            # ---- Base Yaw 绝对角度跟踪 ----
            tracking_base_yaw = 2.0            # yaw 角度跟踪（防止朝向漂移）

            # ---- 惩罚项 ----
            # 注意：contact_no_vel 已删除（无对应 reward 函数实现，不生效）
            feet_contact_forces = -0.01
            # 反抖动：调回合理量级（原 5x/4x 过重，让 policy 不敢出拳）
            action_rate = -0.1                # 原 -0.5 → -0.1
            action_smoothness = -0.05         # 原 -0.2 → -0.05
            dof_acc = -2.5e-07
            energy_cost = -0.001
            collision = -2.0
            dof_pos_limits = -5.0
            dof_vel_limits = -5.0

        # ---- 拳击动作专用 sigma（放宽 arm/yaw/feet, 避免政策崩坏奖励）----
        class sigma_overrides:
            tracking_arm_joint_pos = 1.5      # 原 0.4 → 1.5：boxing arm 动作幅度大(1-2 rad)，σ 应放宽
            tracking_arm_joint_vel = 20.0
            tracking_arm_max_joint_pos = 2.0   # 原 1.0 → 2.0：放宽
            tracking_leg_joint_pos = 1.5
            tracking_leg_joint_vel = 20.0
            tracking_base_yaw = 0.3           # 原 0.08 → 0.3：避免 policy 拿到不到奖励 → 放弃优化 yaw
            # 修复后直接跟踪真正的脚部位置，sigma 适度放宽（避免 0.05 过紧导致 feet 失败后权重恶化）
            tracking_feet_pos = 0.1           # 原 0.05 → 0.1

    # ------------------------------------------------------------------ #
    # domain_rand — 保守设置（拳击平衡敏感）
    # ------------------------------------------------------------------ #
    class domain_rand(N2MimicCfg.domain_rand):
        randomize_gains = True
        p_gain_range = [0.8, 1.2]
        d_gain_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_com_displacement = True
        com_displacement_range = [-0.03, 0.03]   # 缩小：拳击平衡敏感

        randomize_friction = True
        friction_range = [0.5, 1.5]             # 缩小范围：更稳定的地面

        randomize_restitution = True

        randomize_base_mass = True
        added_mass_range = [-3.0, 3.0]          # 缩小：拳击平衡敏感

        # ⚠️ boxing3 关闭扰动 —— boxing 对平衡敏感，外力扰动让 policy 学不到节律性出拳
        disturbance = False
        # 保留字段以兼容 env 访问（push_force_range / push_torque_range 不再生效）
        push_force_range = [30.0, 100.0]
        push_torque_range = [15.0, 50.0]
        disturbance_probabilities = 0.0

    # ------------------------------------------------------------------ #
    # motion_loader
    # ------------------------------------------------------------------ #
    class motion_loader(N2MimicCfg.motion_loader):
        reference_motion_file = [
            "/home/fff/noetix/noetix_n2_gym/datasets/mocap_motions/ning/boxing/Box_3.json",
        ]

    # ------------------------------------------------------------------ #
    # asset — 仅拳击3任务需要的覆盖
    # ------------------------------------------------------------------ #
    class asset(N2MimicCfg.asset):
        penalize_contacts_on = ["base"]  # 手臂/手部拳击时频繁接触地面属正常，仅惩罚躯干碰地

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    class reset:
        root_z_offset = 0.0
        foot_ground_gap = 0.01
        # 过渡阶段时长（与 rewards.transition 保持一致）
        stand_stable_duration_s = 0.3
        startup_duration_s = 1.0

    # ------------------------------------------------------------------ #
    # 覆盖父类 reward 全局参数（反高频抖动策略）
    # ------------------------------------------------------------------ #
    # 关闭 adaptive sigma：防止 robot 通过小幅抖动让 sigma 收紧反而得到更高奖励
    enable_adaptive_tracking_sigma = False
    # 速度软阈值收紧：dof_vel 超过限位 60% 即开始惩罚
    soft_dof_vel_limit = 0.6
    soft_dof_pos_limit = 0.85


class N2Boxing3CfgPPO(N2MimicCfgPPO):
    class runner(N2MimicCfgPPO.runner):
        max_iterations = 50000
        experiment_name = 'n2_boxing3_conservative_ppo'
        save_interval = 200
        resume = False

    class policy(N2MimicCfgPPO.policy):
        init_noise_std = 0.8

    class algorithm(N2MimicCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 3e-4
        desired_kl = 0.005
        num_learning_epochs = 3
