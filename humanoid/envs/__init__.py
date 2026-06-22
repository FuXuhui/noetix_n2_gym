from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from humanoid.utils.task_registry import task_registry

# ---------------------------------------------- Base ----------------------------------------------
from .n2.n2_env import N2Env
from .n2.n2_config import N2_18DofCfg, N2_18DofCfgPPO
from .n2.n2_10dof_env import N2_10dof_Env
from .n2.n2_10dof_config import N2_10dof_Cfg, N2_10dof_CfgPPO


task_registry.register( "n2", N2Env, N2_18DofCfg(), N2_18DofCfgPPO() )
task_registry.register( "n2_10dof", N2_10dof_Env, N2_10dof_Cfg(), N2_10dof_CfgPPO() )

# ---------------------------------------------- Mimic ----------------------------------------------
from .n2.n2_mimic_env import N2MimicEnv
from .n2.n2_mimic_config import N2MimicCfg, N2MimicCfgPPO
task_registry.register( "n2_mimic", N2MimicEnv, N2MimicCfg(), N2MimicCfgPPO() )

# --------------------------------------- Motion Tracking -----------------------------------
from .n2.n2_mimic_env import N2MimicEnv
from .n2.n2_walking3_config import N2Walking3Cfg, N2Walking3CfgPPO
from .n2.n2_boxing3_config import N2Boxing3Cfg, N2Boxing3CfgPPO
task_registry.register( "n2_walking3", N2MimicEnv, N2Walking3Cfg(), N2Walking3CfgPPO() )
task_registry.register( "n2_boxing3", N2MimicEnv, N2Boxing3Cfg(), N2Boxing3CfgPPO() )








