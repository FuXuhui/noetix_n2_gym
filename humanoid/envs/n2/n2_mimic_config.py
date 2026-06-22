import glob
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class N2MimicCfg(LeggedRobotCfg):
    """
    N2人形机器人的运动模仿配置类
    """
    class init_state(LeggedRobotCfg.init_state):
        """初始状态配置"""
        # 机器人初始位置 [x, y, z]
        pos = [0.0, 0.0, 0.75]
        
        # 默认关节角度配置（包含手臂和腿部关节）
        default_joint_angles = {
            # 左臂关节
            "L_arm_shoulder_pitch_joint": 0.,
            "L_arm_shoulder_roll_joint": 0.2,
            "L_arm_shoulder_yaw_joint": 0.,
            "L_arm_elbow_joint": 0.,
            
            # 左腿关节
            "L_leg_hip_yaw_joint": 0.,
            "L_leg_hip_roll_joint": 0.,
            "L_leg_hip_pitch_joint": -0.1495,
            "L_leg_knee_joint": 0.3215,
            "L_leg_ankle_joint": -0.1720,

            # 右臂关节
            "R_arm_shoulder_pitch_joint": 0.,
            "R_arm_shoulder_roll_joint": -0.2,
            "R_arm_shoulder_yaw_joint": 0.,
            "R_arm_elbow_joint": 0.,
            
            # 右腿关节
            "R_leg_hip_yaw_joint": 0.,
            "R_leg_hip_roll_joint": 0.,
            "R_leg_hip_pitch_joint": -0.1495,
            "R_leg_knee_joint": 0.3215,
            "R_leg_ankle_joint": -0.1720,
        }

        # 手臂关节索引（用于拳击等需要手臂优先级跟踪的场景）
        arm_dof_idxs = [0, 1, 2, 3, 9, 10, 11, 12]   # L/R shoulder_pitch/roll/yaw + elbow
        # 腿部关节索引（用于平衡支撑）
        leg_dof_idxs = [4, 5, 6, 7, 8, 13, 14, 15, 16, 17]  # L/R hip_yaw/roll/pitch + knee + ankle

    class env(LeggedRobotCfg.env):
        """环境配置"""
        # 并行环境数量
        num_envs = 2600
        # 帧堆叠数量
        frame_stack = 5
        # 单次观测维度
        num_single_obs = 62 
        # 特权观测维度
        num_privileged_obs = 343
        # 观测空间总维度（帧堆叠后的结果）
        num_observations = int(frame_stack * num_single_obs)
        # 动作空间维度（18个自由度）
        num_actions = 18
        # 回合长度（秒）
        episode_length_s = 45
    
    class termination:
        """终止条件配置"""
        # 是否根据重力方向终止
        terminate_by_gravity = False
        # 是否根据跌倒终止
        terminate_by_fallen = True
        # 当动作结束时是否终止
        terminate_when_motion_end = False
        # 当动作偏差过大时是否终止
        terminate_when_motion_far = True

        class scales:
            """终止条件缩放因子"""
            # 重力方向终止阈值
            termination_gravity = 0.9
            # 动作偏差终止阈值
            termination_motion_far_threshold = 1.5
            # 高度终止阈值
            termination_height = 0.4

        class termination_curriculum:
            """终止条件课程学习配置"""
            # 是否启用动作偏差终止的课程学习
            terminate_when_motion_far_curriculum = True
            # 初始动作偏差终止阈值
            terminate_when_motion_far_initial_threshold = 1.5
            # 最大动作偏差终止阈值
            terminate_when_motion_far_threshold_max = 2.0
            # 最小动作偏差终止阈值
            terminate_when_motion_far_threshold_min = 0.3
            # 动作偏差课程学习速率
            terminate_when_motion_far_curriculum_degree = 2.5e-05
            # 降低课程等级的阈值
            terminate_when_motion_far_curriculum_level_down_threshold = 40
            # 提升课程等级的阈值
            terminate_when_motion_far_curriculum_level_up_threshold = 42
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        """域随机化配置"""
        # 是否随机化增益参数
        randomize_gains = True
        # P增益范围
        p_gain_range = [0.8, 1.2]
        # D增益范围
        d_gain_range = [0.8, 1.2]

        # 是否随机化电机强度
        randomize_motor_strength = True
        # 电机强度范围
        motor_strength_range = [0.8, 1.2]

        # 是否随机化质心偏移
        randomize_com_displacement = True
        # 质心偏移范围
        com_displacement_range = [-0.05, 0.05]

        # 是否随机化摩擦系数
        randomize_friction = True
        # 摩擦系数范围
        friction_range = [0.1, 2.]

        # 是否随机化恢复系数
        randomize_restitution = True
        # 恢复系数范围
        restitution_range = [0., 1.]

        # 是否随机化基础质量
        randomize_base_mass = True
        # 添加质量范围
        added_mass_range = [-5., 5.]

        # 是否启用扰动
        disturbance = True
        # 推力范围
        push_force_range = [50.0, 300.0]
        # 推扭矩范围
        push_torque_range = [25.0, 100.0]
        # 扰动概率
        disturbance_probabilities = 0.002
        # 扰动间隔（单位：dt * decimation 毫秒）
        disturbance_interval = [10, 25] # * dt * decimation ms
    
    class control(LeggedRobotCfg.control):
        """控制配置"""
        # PD控制器参数:
        # 控制类型为位置控制
        control_type = 'P'
        
        # 包含手臂和腿部关节的刚度参数
        stiffness = {
            # 手臂关节刚度
            "arm_shoulder_pitch_joint": 30.0, 
            "arm_shoulder_roll_joint": 30.0, 
            "arm_shoulder_yaw_joint": 30.0, 
            "arm_elbow_joint": 30.0,
            
            # 腿部关节刚度
            'leg_hip_yaw_joint': 80.0, 
            'leg_hip_roll_joint': 80.0, 
            'leg_hip_pitch_joint': 120.0,
            'leg_knee_joint': 120.0, 
            'leg_ankle_joint': 20.0
        }
        
        # 包含手臂和腿部关节的阻尼参数
        damping = {
            # 手臂关节阻尼
            "arm_shoulder_pitch_joint": 1.0, 
            "arm_shoulder_roll_joint": 1.0, 
            "arm_shoulder_yaw_joint": 1.0, 
            "arm_elbow_joint": 1.0,
            
            # 腿部关节阻尼
            'leg_hip_yaw_joint': 5.0, 
            'leg_hip_roll_joint': 5.0, 
            'leg_hip_pitch_joint': 5.0,
            'leg_knee_joint': 5.0, 
            'leg_ankle_joint': 2.0
        }
        
        # 动作缩放因子: 目标角度 = actionScale * action + defaultAngle
        action_scale = 0.25
        # 降采样: 每个策略DT中的控制动作更新次数 @ 仿真DT
        decimation = 10 
    
    class sim(LeggedRobotCfg.sim):
        """仿真配置"""
        # 仿真时间步长
        dt =  0.002

    class asset(LeggedRobotCfg.asset):
        """资产配置"""
        # URDF文件路径
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/N2/urdf/N2.urdf'
        # 机器人名称
        name = "Ning"
        # 足部链接名称
        foot_name = "ankle"
        # 关键部位名称
        key_name = [""]
        # 上身部位名称
        upper_body_name = ["arm"]
        # 下身部位名称
        lower_body_name = ["leg"]
        # 对哪些部位的接触进行惩罚
        penalize_contacts_on = ["base", "knee", "hip", "hand", "arm"]
        # 在哪些部位接触时终止仿真（空列表表示不终止）
        terminate_after_contacts_on = []
        # 自碰撞检测设置（1表示禁用，0表示启用）
        self_collisions = 0  
        # 是否翻转视觉附件
        flip_visual_attachments = False
        # 是否用胶囊体替换圆柱体
        replace_cylinder_with_capsule = False
        # 是否固定基座链接
        fix_base_link = False
    
    class terrain(LeggedRobotCfg.terrain):
        """地形配置"""
        # 网格类型（平面或三角网格）
        mesh_type = 'plane' # plane trimesh
        # 是否启用课程学习
        curriculum = False
        # 仅用于复杂地形:
        # 是否测量高度
        measure_heights = True
        # X方向测量点
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 1mx1.6m rectangle (without center line)
        # Y方向测量点
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
        # 静摩擦系数
        static_friction = 0.6
        # 动摩擦系数
        dynamic_friction = 0.6
        # 地形长度
        terrain_length = 4.
        # 地形宽度
        terrain_width = 4.

        # 地形行数（级别）
        num_rows = 10  
        # 地形列数（类型）
        num_cols = 10  
        # 初始地形等级
        max_init_terrain_level = 0 #10  
        # 地形比例分布 [平面; 障碍物; 均匀; 上坡; 下坡, 上楼梯, 下楼梯]
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # 注释掉的其他地形比例配置
        # terrain_proportions = [0.6, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0]
        # terrain_proportions = [0.3, 0.1, 0.0, 0.2, 0.2, 0.1, 0.1]
        # 恢复系数
        restitution = 0.
    
    class rewards:
        """奖励配置"""
        # 关节位置软限制
        soft_dof_pos_limit = 0.90
        # 关节速度软限制
        soft_dof_vel_limit = 0.80
        # 基础高度目标
        base_height_target = 0.70
        # 最大接触力（超过此值会被惩罚）
        max_contact_force = 400. 
        # 是否只允许正奖励（如果为True，则负总奖励会被截断为零）
        only_positive_rewards = False 
        # 是否使用向量奖励
        use_vec_reward = False

        # 是否启用自适应跟踪sigma
        enable_adaptive_tracking_sigma = True
        # 跟踪sigma的alpha参数
        tracking_sigma_alpha = 0.001
        # 跟踪sigma的缩放因子
        tracking_sigma_scale = 1.0
        # 跟踪sigma的类型
        tracking_sigma_type = "origin"
        # 各种跟踪奖励的sigma值
        reward_tracking_sigma = {
            "tracking_upper_body_pos": 0.015,
            "tracking_lower_body_pos": 0.015,
            "tracking_body_vel": 1.0,
            "tracking_body_ang_vel": 15.0,
            "tracking_feet_pos": 0.01,
            "tracking_max_joint_pos": 1.0,
            "tracking_joint_pos": 0.3,
            "tracking_joint_vel": 30.0,
            # Boxing3: arm joint tracking (arm DOFs separated for priority)
            "tracking_arm_joint_pos": 0.8,
            "tracking_arm_joint_vel": 30.0,
            "tracking_arm_max_joint_pos": 2.0,
            # Boxing3: leg joint tracking (balance support)
            "tracking_leg_joint_pos": 0.5,
            "tracking_leg_joint_vel": 20.0,
        }
        
        # 奖励惩罚课程学习配置
        reward_penalty_curriculum = True
        # 初始惩罚缩放因子
        reward_initial_penalty_scale = 0.1
        # 最小惩罚缩放因子
        reward_min_penalty_scale = 0.0
        # 最大惩罚缩放因子
        reward_max_penalty_scale = 1.0
        # 降低课程等级的阈值
        reward_penalty_level_down_threshold = 40
        # 提升课程等级的阈值
        reward_penalty_level_up_threshold = 42
        # 惩罚课程学习速率
        reward_penalty_degree = 1.0e-05
        # 需要进行惩罚课程学习的奖励名称
        reward_penalty_reward_names = [
            "contact_no_vel",
            "feet_contact_forces",
            "dof_acc",
            "energy_cost",
            "action_smoothness",
            "action_rate",
            "dof_pos_limits",
            "dof_vel_limits"
        ]

        # 计算平均EPL的数量
        num_compute_average_epl = 500  # EMA 平滑（现主要由 current_mean_episode_length 驱动）

        class scales:
            """奖励缩放因子"""
            ################ 奖励 ################
            # 运动跟踪奖励
            tracking_body_pos = 1.0
            tracking_body_vel = 0.5
            tracking_body_ang_vel = 0.5
            tracking_feet_pos = 1.5
            tracking_max_joint_pos = 1.0
            tracking_joint_pos = 1.0
            tracking_joint_vel = 1.0
            tracking_contact_mask = 0.5
            # 其他奖励
            feet_air_time = 1.0
            
            ################ 惩罚 ################
            # 接触惩罚
            contact_no_vel = -5.0
            feet_contact_forces = -0.01
            # 能耗惩罚
            dof_acc = -2.5e-7
            energy_cost = -1e-3
            action_smoothness = -0.05
            action_rate = -0.1
            # 其他惩罚
            collision = -10.0
            dof_pos_limits = -5.0
            dof_vel_limits = -5.0
        

    class noise:
        """噪声配置"""
        # 是否添加噪声
        add_noise = True
        # 噪声等级（1表示按其他值缩放）
        noise_level = 1.0    

        class noise_scales:
            """各类噪声的缩放因子"""
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.2
            lin_vel = 0.05
            gravity = 0.05
            quat = 0.05
            height_measurements = 0.1

    class normalization:
        """归一化配置"""
        class obs_scales:
            """观测值缩放因子"""
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        # 观测值裁剪阈值
        clip_observations = 18.
        # 动作裁剪阈值
        clip_actions = 18.

    class motion_loader:
        """动作加载器配置"""
        # 动作加载器名称
        motion_loader_name = "MotionLoaderNingTracking"
        # 参考动作文件路径（注释掉的是相对路径版本）
        reference_motion_file  = glob.glob("datasets/mocap_motions/ning/dancing/*")
        # 预加载转换数量
        num_preload_transitions = 10
        # 参考观测时间范围
        reference_observation_horizon = 2

class N2MimicCfgPPO(LeggedRobotCfgPPO):
    """N2机器人运动模仿的PPO算法配置类"""
    # 运行器类名（注释掉的是另一种配置）
    # runner_class_name = 'MHOnPolicyRunner' 
    runner_class_name = 'OnPolicyRunner' 
    
    class policy:
        """策略网络配置"""
        # 类名（注释掉的是另一种配置）
        # class_name = 'HIMActorCritic'
        class_name = 'ActorCritic'

        # Actor网络隐藏层维度（注释掉的是另一种配置）
        actor_hidden_dims = [1024, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        # actor_hidden_dims = [256, 256, 256]
        # critic_hidden_dims = [256, 256, 256]
        
        # 初始化噪声标准差
        init_noise_std = 1.0
        # 激活函数（可以是 elu, relu, selu, crelu, lrelu, tanh, sigmoid）
        activation = 'elu' 

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        """算法配置"""
        # 类名（注释掉的是另一种配置）
        # class_name = "MH_PPO" 
        class_name = "PPO" 
        # 熵系数
        entropy_coef = 0.005

        
    class runner( LeggedRobotCfgPPO.runner ):
        """运行器配置"""
        # 最大迭代次数
        max_iterations = 100000
        # 保存间隔
        save_interval = 200 
        # 运行名称
        run_name = ''
        # 实验名称
        experiment_name = 'n2_mimic'
        # 是否在随机回合长度初始化
        init_at_random_ep_len = True