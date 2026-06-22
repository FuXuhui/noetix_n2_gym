import glob
from .n2_mimic_config import N2MimicCfg, N2MimicCfgPPO


class N2Walking3Cfg(N2MimicCfg):
    """N2 Walking_3 motion imitation config with conservative PPO.

    Conservative PPO experiment A:
    1. Reduce learning_rate to 3e-4
    2. Reduce init_noise_std to 0.5
    3. Tighten desired_kl to 0.005
    4. Reduce num_learning_epochs from 5 to 3
    5. Keep previous reward/domain-rand/task fixes unchanged
    """

    # ------------------------------------------------------------------ #
    # env
    # ------------------------------------------------------------------ #
    class env(N2MimicCfg.env):
        episode_length_s = 20  # restored: was 30

    # ------------------------------------------------------------------ #
    # termination — restore per-foot tracking with wide threshold
    # ------------------------------------------------------------------ #
    class termination(N2MimicCfg.termination):
        class scales(N2MimicCfg.termination.scales):
            termination_motion_far_threshold = 2.5     # 放宽到 2.5（key pos 帧距离）

        class termination_curriculum(N2MimicCfg.termination.termination_curriculum):
            terminate_when_motion_far_initial_threshold = 1.8
            terminate_when_motion_far_threshold_min = 0.8
            terminate_when_motion_far_threshold_max = 2.5
            terminate_when_motion_far_curriculum_level_down_threshold = 200  # 宽松：允许机器人多走几步
            terminate_when_motion_far_curriculum_level_up_threshold = 600   # 良好：收紧难度
            terminate_when_motion_far_curriculum_degree = 3e-3              # 适度快速

    # ------------------------------------------------------------------ #
    # rewards — restore penalty curriculum and contact penalty
    # ------------------------------------------------------------------ #
    class rewards(N2MimicCfg.rewards):
        reward_penalty_curriculum = True
        reward_initial_penalty_scale = 0.1
        reward_penalty_level_down_threshold = 100   # 短 episode：惩罚过重导致崩溃
        reward_penalty_level_up_threshold = 600    # 长 episode：性能良好，逐步增加惩罚
        reward_penalty_degree = 3e-3               # 适度快速响应

        class scales(N2MimicCfg.rewards.scales):
            # Tracking rewards (no tracking_root_pos — world coord conflicts with local tracking)
            tracking_body_pos = 1.0
            tracking_body_vel = 0.5
            tracking_body_ang_vel = 0.5
            tracking_feet_pos = 1.5
            tracking_joint_pos = 1.0
            tracking_joint_vel = 1.0
            tracking_max_joint_pos = 1.0
            tracking_contact_mask = 0.5
            # No tracking_root_pos — root tracking in world coords is NOT in obs

            # Gait rewards
            feet_air_time = 1.5

            # Penalties (restored)
            contact_no_vel = -5.0           # restored: was -0.3
            feet_contact_forces = -0.01
            action_rate = -0.1              # restored: was -0.05
            action_smoothness = -0.05        # restored: was -0.01
            dof_acc = -2.5e-07              # restored
            energy_cost = -0.001             # restored
            collision = -10.0
            dof_pos_limits = -5.0
            dof_vel_limits = -5.0

    # ------------------------------------------------------------------ #
    # domain_rand — restore original wide ranges
    # ------------------------------------------------------------------ #
    class domain_rand(N2MimicCfg.domain_rand):
        randomize_gains = True
        p_gain_range = [0.8, 1.2]          # restored: was [0.9, 1.1]
        d_gain_range = [0.8, 1.2]           # restored: was [0.9, 1.1]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]  # restored: was [0.9, 1.1]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]  # restored: was [-0.02, 0.02]

        randomize_friction = True
        friction_range = [0.1, 2.0]         # restored: was [0.6, 1.4]

        randomize_restitution = True        # restored: was False

        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]      # restored: was [-1.0, 1.0]

        disturbance = True
        push_force_range = [50.0, 300.0]    # restored: was [30.0, 80.0]
        push_torque_range = [25.0, 100.0]   # restored: was [10.0, 30.0]
        disturbance_probabilities = 0.002     # restored: was 0.0005

    # ------------------------------------------------------------------ #
    # motion_loader
    # ------------------------------------------------------------------ #
    class motion_loader(N2MimicCfg.motion_loader):
        reference_motion_file = [
            "/home/fff/noetix/noetix_n2_gym/datasets/mocap_motions/ning/walking/Walking_3.json",
        ]

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    class reset:
        root_z_offset = 0.0
        foot_ground_gap = 0.01


class N2Walking3CfgPPO(N2MimicCfgPPO):
    class runner(N2MimicCfgPPO.runner):
        max_iterations = 50000
        experiment_name = 'n2_walking3_conservative_ppo'
        save_interval = 100
        resume = True
        load_run = '0606_00-31-34_'
        checkpoint = 24700
        reset_curriculum = True   # resume 时重置 curriculum，让新参数从初始值开始生效

    class policy(N2MimicCfgPPO.policy):
        init_noise_std = 0.5

    class algorithm(N2MimicCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 3e-4
        desired_kl = 0.005
        num_learning_epochs = 3
