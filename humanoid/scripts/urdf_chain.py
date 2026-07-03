"""URDF body 链路工具（不依赖 isaacgym/gym，可被任意脚本 import）。"""

import xml.etree.ElementTree as ET
from pathlib import Path


def build_urdf_chain(urdf_path):
    """构建 body_idx → (parent_body_idx, joint_name) 的映射。

    Args:
        urdf_path: .urdf 文件路径

    Returns:
        bodies: list[str]           — URDF body 名（顺序与 Isaac Gym rigid_body 索引一致）
        parent_of: list[int]        — bodies[i] 的 parent body idx（root 为 -1）
        joint_of: list[str | None]  — 连接 bodies[i] 到 parent 的关节名（fixed 也保留）
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    bodies = [c.attrib['name'] for c in root if c.tag == 'link']
    name_to_idx = {n: i for i, n in enumerate(bodies)}

    parent_of = [-1] * len(bodies)
    joint_of = [None] * len(bodies)

    for c in root:
        if c.tag != 'joint':
            continue
        jname = c.attrib['name']
        parent = child = None
        for gc in c:
            if gc.tag == 'parent':
                parent = gc.attrib['link']
            elif gc.tag == 'child':
                child = gc.attrib['link']
        if parent in name_to_idx and child in name_to_idx:
            ci = name_to_idx[child]
            parent_of[ci] = name_to_idx[parent]
            joint_of[ci] = jname

    return bodies, parent_of, joint_of


def get_ancestor_joint_names(target_body_idx, parent_of, joint_of):
    """返回从 base 到 target_body 链路上的所有关节名（包含 fixed）。"""
    chain = []
    cur = target_body_idx
    while parent_of[cur] != -1:
        jname = joint_of[cur]
        if jname is not None:
            chain.append(jname)
        cur = parent_of[cur]
    return chain


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        urdf = sys.argv[1]
    else:
        # 默认 N2 URDF
        from humanoid import LEGGED_GYM_ROOT_DIR
        urdf = str(Path(LEGGED_GYM_ROOT_DIR) / 'resources' / 'robots' / 'N2' / 'urdf' / 'N2.urdf')

    bodies, parent_of, joint_of = build_urdf_chain(urdf)
    print(f'URDF: {urdf}')
    print(f'Total bodies: {len(bodies)}')
    print()
    print(f'{"Idx":>3} | {"Name":<32} | {"Parent":<32} | Joint')
    for i, b in enumerate(bodies):
        p = bodies[parent_of[i]] if parent_of[i] >= 0 else '-'
        j = joint_of[i] or '-'
        print(f'{i:>3} | {b:<32} | {p:<32} | {j}')