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
            terminate_when_motion_far_initial_threshold = 8.0   # 拳击动作幅度大（2+ rad/joint）：初始宽松
            terminate_when_motion_far_threshold_min = 1.5
            terminate_when_motion_far_threshold_max = 10.0
            terminate_when_motion_far_curriculum_level_down_threshold = 30   # ep_len < 30：放宽阈值（允许更长 ep）
            terminate_when_motion_far_curriculum_level_up_threshold = 200   # ep_len > 200：收紧阈值（追求精确跟踪）
            terminate_when_motion_far_curriculum_degree = 0.05   # 快速响应

    # ------------------------------------------------------------------ #
    # rewards — 手臂关节优先级 + 平衡保护
    # ------------------------------------------------------------------ #
    class rewards(N2MimicCfg.rewards):
        motion_warmup_time_s = 0.5   # test 模式：站立热身时间（秒），过渡到动作起点
        reward_penalty_curriculum = True
        reward_initial_penalty_scale = 0.1
        reward_penalty_level_down_threshold = 10   # 极短 ep 不调整
        reward_penalty_level_up_threshold = 200   # 与 terminate_when_motion_far.level_up 对齐
        reward_penalty_degree = 0.01

        class scales(N2MimicCfg.rewards.scales):
            # ---- 身体位置跟踪（整体感知，非按部位分开）----
            tracking_body_pos = 2.5          # 提升：留原地比跟手更基本
            tracking_body_vel = 1.5           # 提升：速度跟踪的梯度更陡
            tracking_body_ang_vel = 1.2      # 提升：旋转稳定性
            tracking_feet_pos = 2.0          # 脚部位置（平衡支撑面）

            # ---- 关节跟踪（arm/leg 分开主导，整体仅作 fallback）----
            tracking_joint_pos = 0.2          # 大幅降低：与 arm/leg 重复，避免双重梯度
            tracking_joint_vel = 0.2          # 同步降低
            tracking_max_joint_pos = 0.5      # 同步降低
            tracking_contact_mask = 0.5       # 接触掩码

            # ---- 拳击专用：手臂关节高精度跟踪（核心）----
            tracking_arm_joint_pos = 4.0     # 提升：手臂关节位置（最重要）
            tracking_arm_joint_vel = 2.5     # 提升：手臂关节速度
            tracking_arm_max_joint_pos = 2.5 # 提升：手臂最大关节偏差

            # ---- 拳击专用：腿部关节跟踪（平衡用，降低权重）----
            tracking_leg_joint_pos = 0.8    # 降低：腿部跟踪让位给平衡
            tracking_leg_joint_vel = 0.5    # 降低

            # ---- 步态/平衡奖励 ----
            feet_air_time = 0.5              # 保持支撑

            # ---- 惩罚项 ----
            contact_no_vel = -5.0
            feet_contact_forces = -0.01
            action_rate = -0.1
            action_smoothness = -0.05
            dof_acc = -2.5e-07
            energy_cost = -0.001
            collision = -2.0
            dof_pos_limits = -5.0
            dof_vel_limits = -5.0

        # ---- 拳击动作专用 sigma（更宽松以适应大动作幅度）----
        class sigma_overrides:
            tracking_arm_joint_pos = 0.6      # 收紧：手臂关节精度更高
            tracking_arm_joint_vel = 30.0
            tracking_arm_max_joint_pos = 1.5  # 收紧
            tracking_leg_joint_pos = 0.7      # 放松：腿部 sigma 更大（容忍更大偏差）
            tracking_leg_joint_vel = 20.0

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

        disturbance = True
        push_force_range = [30.0, 100.0]         # 缩小推力：保护平衡
        push_torque_range = [15.0, 50.0]
        disturbance_probabilities = 0.001

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


class N2Boxing3CfgPPO(N2MimicCfgPPO):
    class runner(N2MimicCfgPPO.runner):
        max_iterations = 50000
        experiment_name = 'n2_boxing3_conservative_ppo'
        save_interval = 100
        resume = False

    class policy(N2MimicCfgPPO.policy):
        init_noise_std = 0.5

    class algorithm(N2MimicCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 3e-4
        desired_kl = 0.005
        num_learning_epochs = 3
