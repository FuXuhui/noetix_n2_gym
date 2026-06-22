#!/usr/bin/env python3
"""Generate static visualization of motion JSON as PNG files."""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# FK utilities (same as before)
URDF_LINK_NAMES = [
    'base_link',
    'L_arm_shoulder_pitch_Link', 'L_arm_shoulder_roll_Link', 'L_arm_shoulder_yaw_Link',
    'L_arm_elbow_Link', 'L_arm_hand_Link',
    'L_leg_hip_yaw_link', 'L_leg_hip_roll_link', 'L_leg_hip_pitch_link',
    'L_leg_knee_link', 'L_leg_ankle_link',
    'R_arm_shoulder_pitch_Link', 'R_arm_shoulder_roll_Link', 'R_arm_shoulder_yaw_Link',
    'R_arm_elbow_Link', 'R_arm_hand_link',
    'R_leg_hip_yaw_link', 'R_leg_hip_roll_link', 'R_leg_hip_pitch_link',
    'R_leg_knee_link', 'R_leg_ankle_link',
]
LINK_TO_IDX = {n: i for i, n in enumerate(URDF_LINK_NAMES)}

URDF_JOINT_DEFS = [
    ('L_arm_shoulder_pitch_joint',  'L_arm_shoulder_pitch_Link',  'base_link',                 'x'),
    ('L_arm_shoulder_roll_joint',  'L_arm_shoulder_roll_Link',  'L_arm_shoulder_pitch_Link', 'z'),
    ('L_arm_shoulder_yaw_joint',    'L_arm_shoulder_yaw_Link',  'L_arm_shoulder_roll_Link',  'y'),
    ('L_arm_elbow_joint',           'L_arm_elbow_Link',          'L_arm_shoulder_yaw_Link',   'y'),
    ('L_arm_hand_joint',            'L_arm_hand_Link',           'L_arm_elbow_Link',          'y'),
    ('L_leg_hip_yaw_joint',        'L_leg_hip_yaw_link',       'base_link',                 'y'),
    ('L_leg_hip_roll_joint',       'L_leg_hip_roll_link',      'L_leg_hip_yaw_link',       'x'),
    ('L_leg_hip_pitch_joint',      'L_leg_hip_pitch_link',     'L_leg_hip_roll_link',      'y'),
    ('L_leg_knee_joint',           'L_leg_knee_link',          'L_leg_hip_pitch_link',     'y'),
    ('L_leg_ankle_joint',          'L_leg_ankle_link',         'L_leg_knee_link',          'y'),
    ('R_arm_shoulder_pitch_joint', 'R_arm_shoulder_pitch_Link', 'base_link',                 'x'),
    ('R_arm_shoulder_roll_joint',  'R_arm_shoulder_roll_Link', 'R_arm_shoulder_pitch_Link', 'z'),
    ('R_arm_shoulder_yaw_joint',    'R_arm_shoulder_yaw_Link',  'R_arm_shoulder_roll_Link',  'y'),
    ('R_arm_elbow_joint',           'R_arm_elbow_Link',          'R_arm_shoulder_yaw_Link',   'y'),
    ('R_arm_hand_joint',            'R_arm_hand_link',           'R_arm_elbow_Link',          'y'),
    ('R_leg_hip_yaw_joint',        'R_leg_hip_yaw_link',       'base_link',                 'y'),
    ('R_leg_hip_roll_joint',       'R_leg_hip_roll_link',      'R_leg_hip_yaw_link',       'x'),
    ('R_leg_hip_pitch_joint',      'R_leg_hip_pitch_link',     'R_leg_hip_roll_link',      'y'),
    ('R_leg_knee_joint',           'R_leg_knee_link',          'R_leg_hip_pitch_link',     'y'),
    ('R_leg_ankle_joint',          'R_leg_ankle_link',         'R_leg_knee_link',          'y'),
]
_link_parents = {}
for jname, child, parent, axis in URDF_JOINT_DEFS:
    _link_parents[child] = (parent, axis)

LINK_ORIGINS = {
    'base_link':                  np.array([0.0,  0.0,  0.0]),
    'L_arm_shoulder_pitch_Link':  np.array([0.0, -0.1,  0.0]),
    'L_arm_shoulder_roll_Link':   np.array([0.0,  0.0,  0.0]),
    'L_arm_shoulder_yaw_Link':     np.array([0.0,  0.0,  0.0]),
    'L_arm_elbow_Link':           np.array([0.0,  0.0, -0.1]),
    'L_arm_hand_Link':            np.array([0.0,  0.0, -0.05]),
    'L_leg_hip_yaw_link':        np.array([0.0,  0.0, -0.05]),
    'L_leg_hip_roll_link':       np.array([0.0,  0.0, -0.1]),
    'L_leg_hip_pitch_link':      np.array([0.0,  0.0, -0.2]),
    'L_leg_knee_link':           np.array([0.0,  0.0, -0.2]),
    'L_leg_ankle_link':          np.array([0.0,  0.0, -0.05]),
    'R_arm_shoulder_pitch_Link':  np.array([0.0,  0.1,  0.0]),
    'R_arm_shoulder_roll_Link':   np.array([0.0,  0.0,  0.0]),
    'R_arm_shoulder_yaw_Link':     np.array([0.0,  0.0,  0.0]),
    'R_arm_elbow_Link':           np.array([0.0,  0.0, -0.1]),
    'R_arm_hand_link':            np.array([0.0,  0.0, -0.05]),
    'R_leg_hip_yaw_link':        np.array([0.0,  0.0, -0.05]),
    'R_leg_hip_roll_link':       np.array([0.0,  0.0, -0.1]),
    'R_leg_hip_pitch_link':      np.array([0.0,  0.0, -0.2]),
    'R_leg_knee_link':           np.array([0.0,  0.0, -0.2]),
    'R_leg_ankle_link':          np.array([0.0,  0.0, -0.05]),
}

JOINT_AXES = {'x': np.array([1.,0.,0.]), 'y': np.array([0.,1.,0.]), 'z': np.array([0.,0.,1.])}


def rotmat_from_quat(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*x-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])


def rodrigues(axis, angle):
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def compute_fk(root_pos, root_quat, dof_pos):
    R_root = rotmat_from_quat(root_quat)
    t_root = root_pos.copy()
    world_positions = np.zeros((21, 3))
    world_positions[LINK_TO_IDX['base_link']] = np.zeros(3)

    def get_angle(def_idx):
        return 0.0 if def_idx in [4, 9, 14, 19] else dof_pos[def_idx]

    def fk(link, R_par, t_par):
        idx = LINK_TO_IDX[link]
        if world_positions[idx].sum() != 0:
            return
        if link == 'base_link':
            world_positions[idx] = np.zeros(3)
        else:
            def_idx = next((di for di, d in enumerate(URDF_JOINT_DEFS) if d[1] == link), -1)
            if def_idx < 0: return
            angle = get_angle(def_idx)
            parent_name, axis = _link_parents[link]
            R_j = rodrigues(JOINT_AXES[axis], angle)
            R_link = R_par @ R_j
            t_link = R_par @ LINK_ORIGINS[link] + t_par
            world_positions[idx] = t_link
            for di, d in enumerate(URDF_JOINT_DEFS):
                if d[2] == link: fk(d[1], R_link, t_link)

    fk('base_link', R_root, np.zeros(3))
    for i, n in enumerate(URDF_LINK_NAMES):
        if world_positions[i].sum() == 0:
            world_positions[i] = np.zeros(3)

    positions_local = np.zeros_like(world_positions)
    positions_local[LINK_TO_IDX['base_link']] = np.zeros(3)
    for i in range(21):
        if i != LINK_TO_IDX['base_link']:
            positions_local[i] = R_root.T @ (world_positions[i] - t_root)
    return positions_local


SKELETON_CONNECTIONS = [
    ('base_link', 'L_arm_shoulder_pitch_Link'), ('base_link', 'R_arm_shoulder_pitch_Link'),
    ('base_link', 'L_leg_hip_yaw_link'), ('base_link', 'R_leg_hip_yaw_link'),
    ('L_arm_shoulder_pitch_Link', 'L_arm_shoulder_roll_Link'),
    ('L_arm_shoulder_roll_Link', 'L_arm_shoulder_yaw_Link'),
    ('L_arm_shoulder_yaw_Link', 'L_arm_elbow_Link'),
    ('L_arm_elbow_Link', 'L_arm_hand_Link'),
    ('R_arm_shoulder_pitch_Link', 'R_arm_shoulder_roll_Link'),
    ('R_arm_shoulder_roll_Link', 'R_arm_shoulder_yaw_Link'),
    ('R_arm_shoulder_yaw_Link', 'R_arm_elbow_Link'),
    ('R_arm_elbow_Link', 'R_arm_hand_link'),
    ('L_leg_hip_yaw_link', 'L_leg_hip_roll_link'),
    ('L_leg_hip_roll_link', 'L_leg_hip_pitch_link'),
    ('L_leg_hip_pitch_link', 'L_leg_knee_link'),
    ('L_leg_knee_link', 'L_leg_ankle_link'),
    ('R_leg_hip_yaw_link', 'R_leg_hip_roll_link'),
    ('R_leg_hip_roll_link', 'R_leg_hip_pitch_link'),
    ('R_leg_hip_pitch_link', 'R_leg_knee_link'),
    ('R_leg_knee_link', 'R_leg_ankle_link'),
]


def draw_skeleton(ax, positions, color='steelblue', alpha=1.0, lw=3):
    pts = {n: positions[LINK_TO_IDX[n]] for n in LINK_TO_IDX}
    for a, b in SKELETON_CONNECTIONS:
        if a not in pts or b not in pts: continue
        pa, pb = pts[a], pts[b]
        ax.plot([pa[0],pb[0]], [pa[1],pb[1]], [pa[2],pb[2]],
                color=color, linewidth=lw, alpha=alpha)
    xs, ys, zs = [], [], []
    for n in LINK_TO_IDX:
        xs.append(positions[LINK_TO_IDX[n], 0])
        ys.append(positions[LINK_TO_IDX[n], 1])
        zs.append(positions[LINK_TO_IDX[n], 2])
    ax.scatter(xs, ys, zs, color=color, s=50, alpha=alpha, zorder=10)
    for foot in ['L_leg_ankle_link', 'R_leg_ankle_link']:
        f = pts[foot]
        ax.scatter([f[0]],[f[1]],[f[2]], color='red', s=80, alpha=alpha, zorder=11)


def get_frame_positions(frames, indices):
    positions_list = []
    for i in indices:
        f = frames[i]
        root_pos = np.array(f[:3])
        root_quat = np.array(f[3:7])
        dof_pos = np.array(f[7:25])
        positions_list.append(compute_fk(root_pos, root_quat, dof_pos))
    return np.array(positions_list)


def plot_multiview(json_path, output_png):
    with open(json_path) as f:
        md = json.load(f)

    frames = md['Frames']
    n = len(frames)
    dt = md['FrameDuration']
    motion_name = json_path.split('/')[-1].replace('.json', '')

    # Choose 6 key frames evenly distributed
    indices = list(np.linspace(0, n-1, 6, dtype=int))

    pos = get_frame_positions(frames, indices)

    # Compute axis limits
    all_x = pos[:,:,0].flatten()
    all_y = pos[:,:,1].flatten()
    all_z = pos[:,:,2].flatten()
    margin = 0.15
    xlim = [all_x.min()-margin, all_x.max()+margin]
    ylim = [all_y.min()-margin, all_y.max()+margin]
    zlim = [all_z.min()-margin, all_z.max()+margin]
    zmid = (zlim[0]+zlim[1])/2

    fig = plt.figure(figsize=(16, 18))
    fig.patch.set_facecolor('#1a1a2e')
    views = [(30, 45), (30, 135), (0, 90), (90, 0)]
    view_names = ['3D Isometric', '3D Opposite', 'Front (X-Y)', 'Top (X-Z)']

    for row, fi in enumerate(indices):
        positions = pos[row]
        for col, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(len(indices), len(views), row * len(views) + col + 1, projection='3d')
            ax.patch.set_facecolor('#1a1a2e')
            draw_skeleton(ax, positions, color='steelblue', alpha=0.9, lw=3)
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
            ax.set_box_aspect([1,1,1.5])
            ax.view_init(elev=elev, azim=azim)
            ax.axis('off')
            if col == 0:
                ax.text2D(0.02, 0.98, f't={fi * dt:.1f}s', transform=ax.transAxes,
                          color='white', fontsize=9, va='top')
            if row == 0:
                ax.set_title(view_names[col], color='white', fontsize=10)

    plt.suptitle(f'{motion_name}  |  {n} frames @ {1/dt:.1f} fps  |  Duration: {n*dt:.1f}s',
                 color='white', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_png, dpi=120, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print(f'Saved: {output_png}')
    plt.close()


def plot_joint_time_series(json_path, output_png):
    with open(json_path) as f:
        md = json.load(f)

    frames = md['Frames']
    n = len(frames)
    dt = md['FrameDuration']
    t = np.arange(n) * dt
    dof_pos = np.array([f[7:25] for f in frames])

    joint_names = [
        'L_arm_shoulder_pitch', 'L_arm_shoulder_roll', 'L_arm_shoulder_yaw', 'L_arm_elbow',
        'L_leg_hip_yaw', 'L_leg_hip_roll', 'L_leg_hip_pitch', 'L_leg_knee',
        'R_arm_shoulder_pitch', 'R_arm_shoulder_roll', 'R_arm_shoulder_yaw', 'R_arm_elbow',
        'R_leg_hip_yaw', 'R_leg_hip_roll', 'R_leg_hip_pitch', 'R_leg_knee',
    ]

    fig, axes = plt.subplots(4, 4, figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a2e')
    colors = plt.cm.coolwarm(np.linspace(0, 1, 16))

    for i, (ax, name) in enumerate(zip(axes.flat, joint_names)):
        ax.set_facecolor('#2d2d44')
        ax.plot(t, dof_pos[:, i], color=colors[i], linewidth=1.5)
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
        ax.set_title(name, color='white', fontsize=9, pad=3)
        ax.tick_params(colors='white', labelsize=7)
        ax.grid(True, alpha=0.15, color='white')
        if i % 4 == 0:
            ax.set_ylabel('rad', color='white', fontsize=7)
        if i >= 12:
            ax.set_xlabel('Time (s)', color='white', fontsize=7)

    motion_name = json_path.split('/')[-1].replace('.json', '')
    plt.suptitle(f'Joint Angles Over Time - {motion_name}', color='white', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_png, dpi=120, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print(f'Saved: {output_png}')
    plt.close()


def plot_root_trajectory(json_path, output_png):
    with open(json_path) as f:
        md = json.load(f)

    frames = md['Frames']
    n = len(frames)
    dt = md['FrameDuration']
    t = np.arange(n) * dt

    root_pos = np.array([f[:3] for f in frames])

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')

    ax1 = fig.add_subplot(131)
    ax1.set_facecolor('#2d2d44')
    ax1.plot(root_pos[:, 0], root_pos[:, 1], 'steelblue', linewidth=1.5)
    ax1.scatter([root_pos[0, 0]], [root_pos[0, 1]], color='green', s=100, zorder=10, label='Start')
    ax1.scatter([root_pos[-1, 0]], [root_pos[-1, 1]], color='red', s=100, zorder=10, label='End')
    ax1.set_xlabel('X (m)', color='white')
    ax1.set_ylabel('Y (m)', color='white')
    ax1.set_title('X-Y Trajectory', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#2d2d44', labelcolor='white', fontsize=8)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.15, color='white')

    ax2 = fig.add_subplot(132)
    ax2.set_facecolor('#2d2d44')
    ax2.plot(t, root_pos[:, 0], label='X', linewidth=1.5)
    ax2.plot(t, root_pos[:, 1], label='Y', linewidth=1.5)
    ax2.plot(t, root_pos[:, 2], label='Z (height)', linewidth=1.5)
    ax2.set_xlabel('Time (s)', color='white')
    ax2.set_ylabel('Position (m)', color='white')
    ax2.set_title('Root Position vs Time', color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#2d2d44', labelcolor='white', fontsize=8)
    ax2.grid(True, alpha=0.15, color='white')

    # Height histogram
    ax3 = fig.add_subplot(133)
    ax3.set_facecolor('#2d2d44')
    ax3.hist(root_pos[:, 2], bins=30, color='steelblue', alpha=0.8)
    ax3.set_xlabel('Height (m)', color='white')
    ax3.set_ylabel('Count', color='white')
    ax3.set_title(f'Height Distribution\nmean={root_pos[:,2].mean():.3f}m', color='white')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.15, color='white')

    motion_name = json_path.split('/')[-1].replace('.json', '')
    plt.suptitle(f'Root Trajectory - {motion_name}', color='white', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_png, dpi=120, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print(f'Saved: {output_png}')
    plt.close()


if __name__ == '__main__':
    import os
    base = '/home/fff/noetix/noetix_n2_gym/datasets/mocap_motions/ning'

    motions = [
        f'{base}/walking/Walking_3.json',
        f'{base}/boxing/Box_3.json',
    ]

    for json_path in motions:
        name = json_path.split('/')[-1].replace('.json', '')
        out_dir = os.path.dirname(json_path)

        print(f'\n--- {name} ---')
        plot_multiview(json_path, f'{out_dir}/{name}_skeleton.png')
        plot_joint_time_series(json_path, f'{out_dir}/{name}_joints.png')
        plot_root_trajectory(json_path, f'{out_dir}/{name}_trajectory.png')
