"""依次驱动 N2 URDF 的 21 个 body，观察每个 body idx 对应的物理实体。

目的
----
验证 URDF idx 与 ref motion body idx 的对应关系（详见 docs/urdf_body_mapping.md）。

用法
----
# 对单个 body 驱动 10s（headless）
python humanoid/scripts/sweep_bodies.py --body 15 --duration 10

# 一次性跑完所有 21 个 body，每个 5s（总 ~105s）
python humanoid/scripts/sweep_bodies.py --all --duration 5

# 输出 JSON 报告
python humanoid/scripts/sweep_bodies.py --body 20 --duration 8 --output report_body20.json
"""

import argparse
import json
import sys
from pathlib import Path

# 先解析 CLI，避免 isaacgym 初始化前 argparse 抛错
def _early_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--body', type=int, default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--duration', type=float, default=10.0)
    p.add_argument('--amp', type=float, default=0.6)
    p.add_argument('--freq', type=float, default=0.5)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--task', type=str, default='n2_boxing3')
    p.add_argument('--headless', action='store_true')
    return p.parse_known_args()


def main():
    early_args, remaining = _early_parse()

    # 验证 URDF 链路（直接读 xml，不依赖 humanoid package，避免循环导入）
    # scripts 目录的 sys.path 注入可能导致 humanoid package 解析失败
    # 因此 urdf_chain.py 写成纯标准库 + 可选路径解析
    import importlib.util
    _urdf_chain_path = Path(__file__).parent / 'urdf_chain.py'
    _spec = importlib.util.spec_from_file_location('_urdf_chain', _urdf_chain_path)
    _urdf_chain = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_urdf_chain)
    build_urdf_chain = _urdf_chain.build_urdf_chain
    get_ancestor_joint_names = _urdf_chain.get_ancestor_joint_names

    # 现在可以安全 import isaacgym + humanoid
    # 注意：必须先 import humanoid.envs，再 import humanoid.utils；
    # 否则 utils/__init__.py 中的 task_registry 会触发循环导入
    # （play.py 也是同样的顺序）
    from humanoid import LEGGED_GYM_ROOT_DIR  # noqa: 必须在 isaacgym 之前
    import isaacgym  # noqa
    from isaacgym import gymtorch
    import torch

    from humanoid.envs import *  # noqa: 必须先于 humanoid.utils
    from humanoid.utils import get_args, task_registry

    # 加载 URDF 链路
    urdf_path = Path(LEGGED_GYM_ROOT_DIR) / 'resources' / 'robots' / 'N2' / 'urdf' / 'N2.urdf'
    bodies, parent_of, joint_of = build_urdf_chain(urdf_path)

    # 替换 sys.argv 让 task_registry 拿到我们的剩余参数
    sys.argv = [sys.argv[0]] + remaining + ['--headless']
    task_args = get_args()

    # 构造环境
    env_cfg, train_cfg = task_registry.get_cfgs(name=early_args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_friction = False

    env, _ = task_registry.make_env(name=early_args.task, args=task_args, env_cfg=env_cfg)

    # dof name → idx
    name_to_dof_idx = {n: i for i, n in enumerate(env.dof_names)}

    print(f'[init] URDF has {len(bodies)} bodies, env has {len(env.dof_names)} dofs')
    print('[init] bodies (URDF order):')
    for i, b in enumerate(bodies):
        marker = ''
        if parent_of[i] != -1 and joint_of[i] in name_to_dof_idx:
            marker = f'  ← dof="{joint_of[i]}"'
        elif joint_of[i] is not None and 'hand' in joint_of[i]:
            marker = '  (fixed)'
        print(f'   {i:2d}: {b}{marker}')

    dt = env.dt
    print(f'[init] dt={dt}s')
    env.reset()

    # body 列表
    if early_args.all:
        body_list = list(range(len(bodies)))
    else:
        body_list = [early_args.body]

    # sweep
    all_reports = []
    for bidx in body_list:
        print(f'\n{"="*60}')
        print(f'>>> Sweep body[{bidx}] = {bodies[bidx]}')
        print(f'{"="*60}')

        chain_joints = get_ancestor_joint_names(bidx, parent_of, joint_of)
        chain = [name_to_dof_idx[j] for j in chain_joints if j in name_to_dof_idx]

        if not chain:
            print(f'[skip] body[{bidx}]={bodies[bidx]} 无 dof 链路（root 或 fixed）')
            continue

        chain_names = [env.dof_names[i] for i in chain]
        print(f'[drive] body[{bidx}]={bodies[bidx]} → driving {len(chain)} dofs: {chain_names}')

        n_steps = int(early_args.duration / dt)
        samples = []

        # 缓存初始位置
        init_dof_pos = env.dof_pos[0].clone()
        init_root_pos = env.root_states[0, :3].clone()

        for step in range(n_steps):
            t = step * dt
            omega = 2 * 3.14159265 * early_args.freq
            new_dof_pos = init_dof_pos.clone()
            for k, dof_idx in enumerate(chain):
                phase = (k % 2) * 3.14159265
                new_dof_pos[dof_idx] = init_dof_pos[dof_idx] + early_args.amp * torch.sin(
                    torch.tensor(omega * t + phase, device=env.device)
                )
            env.dof_pos[0] = new_dof_pos
            env.dof_vel[0] = 0.0

            env.gym.set_dof_state_tensor_indexed(
                env.sim,
                gymtorch.unwrap_tensor(env.dof_state),
                gymtorch.unwrap_tensor(torch.tensor([0], dtype=torch.int32, device=env.device)),
                1,
            )

            # 推进一步（保持 zero action，让 P 控制不会引入额外干扰）
            env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))

            env.gym.refresh_rigid_body_state_tensor(env.sim)
            body_pos_world = env.rigid_body_states_view[0, :, :3]
            body_pos_rel = body_pos_world - body_pos_world[0:1]
            samples.append({
                't': float(t),
                'root_pos': env.root_states[0, :3].cpu().numpy().tolist(),
                'body_pos_world': body_pos_world.cpu().numpy().tolist(),
                'body_pos_rel': body_pos_rel.cpu().numpy().tolist(),
            })

        # 统计每个 body 在驱动期间相对 base 的最大位移
        body_max_disp = []
        for bi in range(len(bodies)):
            max_d = 0.0
            for s in samples:
                rel = s['body_pos_rel'][bi]
                d = (rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2) ** 0.5
                if d > max_d:
                    max_d = d
            body_max_disp.append(max_d)

        root_drift = []
        for s in samples:
            dx = s['root_pos'][0] - samples[0]['root_pos'][0]
            dy = s['root_pos'][1] - samples[0]['root_pos'][1]
            dz = s['root_pos'][2] - samples[0]['root_pos'][2]
            root_drift.append((dx * dx + dy * dy + dz * dz) ** 0.5)
        max_root_drift = max(root_drift) if root_drift else 0.0

        report = {
            'target_body_idx': int(bidx),
            'target_body_name': bodies[bidx],
            'driven_dofs': chain_names,
            'amp_rad': early_args.amp,
            'freq_hz': early_args.freq,
            'duration_s': early_args.duration,
            'body_names': bodies,
            'body_max_disp_rel_to_base': body_max_disp,
            'max_root_drift': max_root_drift,
            'samples': samples,
        }
        all_reports.append(report)

        # 摘要
        print(f'\n--- Summary for body[{bidx}]={bodies[bidx]} ---')
        print(f'{"idx":>3}  {"name":<32}  {"max disp (m)":>14}  {"status"}')
        for i, (name, disp) in enumerate(zip(bodies, body_max_disp)):
            status = ''
            if i == bidx:
                status = '*** TARGET ***'
            elif disp > 0.05:
                status = '<- moved (chain descendant)'
            print(f'{i:>3}  {name:<32}  {disp:>14.4f}  {status}')
        print(f'root drift during sweep: {max_root_drift:.4f} m')

    # 输出
    if early_args.output:
        out_path = Path(early_args.output)
    elif early_args.all:
        out_path = Path(LEGGED_GYM_ROOT_DIR) / 'docs' / 'sweep_bodies_report.json'
    else:
        out_path = Path(LEGGED_GYM_ROOT_DIR) / f'docs/sweep_body{early_args.body}_report.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    print(f'\n[saved] report → {out_path}')


if __name__ == '__main__':
    main()