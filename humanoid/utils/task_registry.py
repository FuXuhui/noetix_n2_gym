import os
from typing import Tuple
from datetime import datetime

from humanoid.algo import *

from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


def _apply_checkpoint_patch(env_cfg, load_run, checkpoint, name):
    """Patch env_cfg obs dims from a checkpoint's train_cfg.json so the env
    is created with dimensions matching the saved policy."""
    import json
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', name)
    try:
        load_run_dir = os.path.join(log_root, load_run)
        saved_path = os.path.join(load_run_dir, 'train_cfg.json')
        if os.path.exists(saved_path):
            with open(saved_path) as f:
                saved = json.load(f)
            saved_env = saved.get('env', {})
            for key in ['num_single_obs', 'num_privileged_obs', 'num_observations']:
                if key in saved_env:
                    obj = env_cfg
                    for part in key.split('.')[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, key.split('.')[-1], saved_env[key])
            return saved.get('runner', {}).get('experiment_name')
    except Exception as e:
        print(f'[task_registry] Could not patch obs dims: {e}')
    return None


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """Creates an environment."""
        if args is None:
            args = get_args()
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")

        # Always get fresh env_cfg from registry to avoid stale dims from a
        # previous session (e.g. play.py with a different checkpoint).
        env_cfg, _ = self.get_cfgs(name)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)

        # Patch obs dims from checkpoint train_cfg.json BEFORE env creation.
        load_run = getattr(args, 'load_run', None)
        checkpoint = getattr(args, 'checkpoint', None)
        if load_run and str(load_run) != '-1' and checkpoint is not None and checkpoint != -1:
            import json, os
            logs_base = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs')
            run_dir = None
            for exp_name in sorted(os.listdir(logs_base)):
                candidate = os.path.join(logs_base, exp_name, str(load_run))
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, f'model_{checkpoint}.pt')):
                    run_dir = candidate
                    break
            if run_dir:
                saved_path = os.path.join(run_dir, 'train_cfg.json')
                with open(saved_path) as f:
                    saved = json.load(f)
                saved_env = saved.get('env', {})
                patched = []
                for key in ['num_single_obs', 'num_privileged_obs', 'num_observations']:
                    if key in saved_env:
                        obj = env_cfg
                        for part in key.split('.')[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, key.split('.')[-1], saved_env[key])
                        patched.append(f'{key}={saved_env[key]}')
                if patched:
                    print(f'[make_env] Patched dims from checkpoint: {", ".join(patched)}')

        set_seed(env_cfg.seed)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(cfg=env_cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         headless=args.headless)
        self.env_cfg_for_wandb = env_cfg
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """Creates the training algorithm."""
        if args is None:
            args = get_args()
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # When loading a checkpoint, find the correct experiment dir and load dims.
        resume = train_cfg.runner.resume
        resolved_log_root = log_root
        saved_env_cfg = {}  # dims from saved checkpoint
        if resume:
            load_run = getattr(args, 'load_run', None) or train_cfg.runner.load_run
            checkpoint = getattr(args, 'checkpoint', None) or train_cfg.runner.checkpoint
            if load_run and str(load_run) != '-1' and checkpoint is not None and checkpoint != -1:
                import json
                logs_base = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs')
                found_exp = None
                found_run_dir = None
                for exp_name in sorted(os.listdir(logs_base)):
                    exp_dir = os.path.join(logs_base, exp_name)
                    candidate = os.path.join(exp_dir, str(load_run))
                    if os.path.isdir(candidate):
                        ckpt_file = os.path.join(candidate, f'model_{checkpoint}.pt')
                        if os.path.exists(ckpt_file):
                            found_exp = exp_name
                            found_run_dir = candidate
                            break
                if found_exp:
                    resolved_log_root = os.path.join(logs_base, found_exp)
                    print(f'[task_registry] Found checkpoint in exp={found_exp}')
                    saved_path = os.path.join(found_run_dir, 'train_cfg.json')
                    with open(saved_path) as f:
                        saved = json.load(f)
                    saved_env_cfg = saved.get('env', {})
                    patched = []
                    for key in ['num_single_obs', 'num_privileged_obs', 'num_observations']:
                        if key in saved_env_cfg:
                            patched.append(f'{key}={saved_env_cfg[key]}')
                    if patched:
                        print(f'[task_registry] Dims from checkpoint: {", ".join(patched)}')

        if resolved_log_root == "default":
            resolved_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(resolved_log_root, datetime.now().strftime('%m%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif resolved_log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(resolved_log_root, datetime.now().strftime('%m%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        train_cfg_dict = class_to_dict(train_cfg)
        env_cfg_dict = class_to_dict(self.env_cfg_for_wandb)

        # Apply checkpoint dims BEFORE creating runner so policy input size matches checkpoint.
        if saved_env_cfg:
            for key in ['num_single_obs', 'num_privileged_obs', 'num_observations']:
                if key in saved_env_cfg:
                    obj = self.env_cfg_for_wandb
                    for part in key.split('.')[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, key.split('.')[-1], saved_env_cfg[key])
            env_cfg_dict = class_to_dict(self.env_cfg_for_wandb)

        all_cfg = {**train_cfg_dict, **env_cfg_dict}
        
        runner_class = eval(train_cfg_dict["runner_class_name"])
        runner = runner_class(env, all_cfg, log_dir, device=args.rl_device)

        if resume:
            resume_path = get_load_path(resolved_log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path, load_optimizer=False)
            if getattr(train_cfg.runner, 'reset_curriculum', False):
                if hasattr(env, 'reset_curriculum'):
                    env.reset_curriculum()
                else:
                    print(f'[task_registry] Warning: env has no reset_curriculum(), skipping')
        return runner, train_cfg


# make global task registry
task_registry = TaskRegistry()
