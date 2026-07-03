# N2 URDF Body 与 Reference Motion Body 索引对应关系

> 适用于 `resources/robots/N2/urdf/N2.urdf` 与 `datasets/mocap_motions/ning/boxing/Box_3.json`
> 最近修订：2026-06-30（修复 `tracking_feet_pos` 时建立）

---

## 1. 背景

`MotionLoaderNingTracking` 加载的 reference motion 使用 **20 个 body 的 FK 数据**（MuJoCo skeleton），
`TAR_TOE_POS_LOCAL_SIZE = 60 = 20 × 3`，索引范围 `[0, 19]`。

但 N2 URDF 的 `link` 数量是 **21 个**，最后一个 body 是 `R_leg_ankle_link`（索引 20），
**这个 body 在 ref 中不存在**，导致直接用 `torch.clamp(urdf_idx, 0, 19)` 会把右脚踝错误地指向 `base_link` (ref idx 19)。

---

## 2. 完整映射表

### 2.1 URDF Body（21 个，Isaac Gym / PyTorch 顺序）

| URDF idx | link 名 | 父 link | 通过关节 | 部位 |
|---:|---|---|---|---|
| 0  | `base_link`                  | (root)        | —                  | 躯干根 |
| 1  | `L_arm_shoulder_pitch_Link`  | `base_link`   | `L_arm_shoulder_pitch_joint` | 左臂 0 |
| 2  | `L_arm_shoulder_roll_Link`   | L_arm_shoulder_pitch_Link | `L_arm_shoulder_roll_joint` | 左臂 1 |
| 3  | `L_arm_shoulder_yaw_Link`    | L_arm_shoulder_roll_Link | `L_arm_shoulder_yaw_joint` | 左臂 2 |
| 4  | `L_arm_elbow_Link`           | L_arm_shoulder_yaw_Link | `L_arm_elbow_joint` | 左臂 3 |
| 5  | `L_arm_hand_Link`            | L_arm_elbow_Link | `L_arm_hand_joint` (fixed) | 左手（fixed） |
| 6  | `R_arm_shoulder_pitch_Link`  | `base_link`   | `R_arm_shoulder_pitch_joint` | 右臂 0 |
| 7  | `R_arm_shoulder_roll_Link`   | R_arm_shoulder_pitch_Link | `R_arm_shoulder_roll_joint` | 右臂 1 |
| 8  | `R_arm_shoulder_yaw_Link`    | R_arm_shoulder_roll_Link | `R_arm_shoulder_yaw_joint` | 右臂 2 |
| 9  | `R_arm_elbow_Link`           | R_arm_shoulder_yaw_Link | `R_arm_elbow_joint` | 右臂 3 |
| 10 | `R_arm_hand_link`            | R_arm_elbow_Link | `R_arm_hand_joint` (fixed) | 右手（fixed） |
| 11 | `L_leg_hip_yaw_link`         | `base_link`   | `L_leg_hip_yaw_joint` | 左腿 0 |
| 12 | `L_leg_hip_roll_link`        | L_leg_hip_yaw_link | `L_leg_hip_roll_joint` | 左腿 1 |
| 13 | `L_leg_hip_pitch_link`       | L_leg_hip_roll_link | `L_leg_hip_pitch_joint` | 左腿 2 |
| 14 | `L_leg_knee_link`            | L_leg_hip_pitch_link | `L_leg_knee_joint` | 左腿 3 |
| 15 | `L_leg_ankle_link`           | L_leg_knee_link | `L_leg_ankle_joint` | **左踝** |
| 16 | `R_leg_hip_yaw_link`         | `base_link`   | `R_leg_hip_yaw_joint` | 右腿 0 |
| 17 | `R_leg_hip_roll_link`        | R_leg_hip_yaw_link | `R_leg_hip_yaw_joint` | 右腿 1 |
| 18 | `R_leg_hip_pitch_link`       | R_leg_hip_roll_link | `R_leg_hip_pitch_joint` | 右腿 2 |
| 19 | `R_leg_knee_link`            | R_leg_hip_pitch_link | `R_leg_knee_joint` | 右腿 3 |
| **20** | **`R_leg_ankle_link`**   | R_leg_knee_link | `R_leg_ankle_joint` | **右踝（URDF 中存在，ref 中不存在）** |

### 2.2 Reference Motion Body（20 个，Box_3.json 第 0 帧 root-relative 位置）

| ref idx | x    | y    | z    | 识别要点 | 对应部位 |
|---:|---:|---:|---:|---|---|
| 0  |  0.0000 |  0.0000 |  0.0000 | 原点 | base / 根节点 |
| 1  |  0.0000 |  0.0000 |  0.0000 | 原点 | base / 根节点（重复） |
| 2  | -0.0245 |  0.1456 | -0.1942 | y>0, z 低 | 左腿中段（hip/knee） |
| 3  | -0.0245 |  0.1456 | -0.1942 | 同 2 | 左腿中段（与 2 同步） |
| 4  | -0.0251 |  0.1673 | **-0.2155** | **z 最低，y>0** | **左脚（Ankle/Toe）** |
| 5  | -0.1127 |  0.2685 | -0.1083 | 远端 | 左臂末端 |
| 6  |  0.0098 |  0.1168 |  0.1421 |  |  |
| 7  |  0.0098 |  0.1168 |  0.1421 |  |  |
| 8  | -0.0655 |  0.1713 |  0.1722 |  |  |
| 9  | -0.1795 |  0.2705 |  0.3032 |  |  |
| 10 | -0.1145 |  0.2689 |  0.5550 | z 很高 | 左手指尖 |
| 11 |  0.0776 | -0.1708 | -0.1568 | y<0, z 低 | 右腿中段 |
| 12 |  0.0776 | -0.1708 | -0.1568 | 同 11 | 右腿中段（与 11 同步） |
| 13 |  0.0869 | **-0.1979** | **-0.1675** | **z 最低，y<0** | **右脚（Ankle/Toe）** |
| 14 |  0.1061 | -0.2544 | -0.0058 |  |  |
| 15 |  0.0653 | -0.0557 |  0.1630 |  |  |
| 16 |  0.0653 | -0.0557 |  0.1630 |  |  |
| 17 | -0.0153 | -0.0613 |  0.2178 |  |  |
| 18 | -0.1597 | -0.0652 |  0.3560 |  |  |
| 19 | -0.1645 | -0.0139 |  0.6109 | z 很高，y≈0 | 右手指尖 |

### 2.3 跨域对应（URDF ↔ ref）

| 概念 | URDF body idx | ref body idx | 关键说明 |
|---|---|---|---|
| root / 躯干 | 0 | 0 / 1 | ref 中 0 和 1 都为原点 |
| 左肩 pitch/roll/yaw/elbow | 1, 2, 3, 4 | 5, 6, 7, 8, 9, 10 | 手臂末端在 ref idx 10 |
| 右肩 pitch/roll/yaw/elbow | 6, 7, 8, 9 | 14, 15, 16, 17, 18, 19 | 手臂末端在 ref idx 19 |
| 左腿 hip yaw/roll/pitch/knee | 11, 12, 13, 14 | 2, 3 | 中段在 ref idx 2-3 |
| 左踝 | **15** | **4** | ✅ 存在对应，z 最低且 y>0 |
| 右腿 hip yaw/roll/pitch/knee | 16, 17, 18, 19 | 11, 12 | 中段在 ref idx 11-12 |
| **右踝** | **20** | **13** | ⚠️ **URDF 有，ref 中需要按 z/y 推断** |

> 关键发现：URDF 中有 21 个 body，ref 中只有 20 个。`R_leg_ankle_link`（URDF idx 20）**没有对应 ref body**。
> 但因为左右对称性，**左踝 ref idx 4** 和 **右踝 ref idx 13** 的位姿应当分别镜像。
> `tracking_feet_pos` 直接硬编码使用 ref idx `[4, 13]`（来自左右对称推断 + z 最低筛选）。

---

## 3. 原 Bug 与修复

### 3.1 旧实现

```python
# n2_mimic_env.py 旧版本
def _reward_tracking_feet_pos(self):
    ref_key_pos = self.ref_key_pos.reshape(self.num_envs, -1, 3)
    num_ref = ref_key_pos.shape[1]                       # 20

    feet = torch.clamp(self.feet_indices, 0, num_ref - 1)  # URDF [15, 20] → [15, 19]

    feet_diff = ref_key_pos[:, feet, :] - self.key_pos_trunc[:, feet, :]
```

问题：

- `self.feet_indices`（URDF 中含 `"ankle"` 的 body）= `[15, 20]`
- clamp 到 `[0, 19]` 后变成 `[15, 19]`
  - `key_pos_trunc[15]`：URDF idx 15 = `L_leg_ankle_link` ✅
  - `key_pos_trunc[19]`：URDF idx 19 = `R_leg_knee_link` ❌（应为右踝）
- 同样，`ref_key_pos[19]` 是右手指尖，**完全错位**

### 3.2 新实现

```python
def _reward_tracking_feet_pos(self):
    ref_key_pos = self.ref_key_pos.reshape(self.num_envs, -1, 3)
    num_ref = ref_key_pos.shape[1]

    # 直接硬编码左右脚在 ref 中的 idx（来自 z 最低 + 左右对称）
    foot_ref_idxs = torch.tensor([4, 13], dtype=torch.long, device=self.device)
    foot_ref_idxs = torch.clamp(foot_ref_idxs, 0, num_ref - 1)

    # 机器人侧：用 21-body 的 rigid_body_states_view（未截断），URDF feet_indices 直接生效
    feet_pos_robot = self.rigid_body_states_view[:, self.feet_indices, :3]
    root_pos = self.root_states[:, :3].unsqueeze(1)
    feet_pos_robot_local = feet_pos_robot - root_pos

    feet_pos_ref = ref_key_pos[:, foot_ref_idxs, :]
    feet_diff = feet_pos_ref - feet_pos_robot_local
    feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
    rew = torch.exp(-feet_dist / self.cfg.rewards.reward_tracking_sigma["tracking_feet_pos"])
```

关键点：

1. **不依赖 URDF→ref 的索引映射**（避免 R_leg_ankle 越界）
2. **机器人侧用 `rigid_body_states_view`**（21-body，包含 R_leg_ankle 真实位置）
3. **ref 侧用硬编码的 `[4, 13]`**（来自左右脚 z 最低的物理推断）

---

## 4. 验证方法

使用 `humanoid/scripts/sweep_bodies.py`（见同仓库）：

```bash
# 依次控制 URDF idx 0-20 的 body（设置其祖先链上的关节角度），
# 观察 21 个 body 的位置变化，确认哪个 body idx 会动。
python humanoid/scripts/sweep_bodies.py --body 15 --duration 10   # 左踝
python humanoid/scripts/sweep_bodies.py --body 20 --duration 10   # 右踝
```

预期：

- 控制 body 15（左踝）：URDF idx 15 动，URDF idx 20 不动
- 控制 body 20（右踝）：URDF idx 20 动，URDF idx 15 不动
- 在 ref 中：左踝对应 ref idx 4，右踝对应 ref idx 13