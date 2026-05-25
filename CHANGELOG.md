# ReactHuman Pipeline — Changelog

## Session: 2026-05-17

---

### 1. 新任务类型：`stair_tumble`

**背景**：新增第 15 种物理任务——物体从楼梯顶部滚/滑下，向观察者方向运动。

#### 1.1 新房间类型 `stair_landing`

**文件**：`asset_library/rooms.yaml`

新增带 7 级楼梯的房间，楼梯几何参数作为 YAML 字段存储：

```yaml
- type: stair_landing
  width: 5.0
  depth: 5.0
  height: 2.80
  stair_n_steps: 7
  stair_rise: 0.18      # 每级踏步高度（米）
  stair_run: 0.28       # 每级踏步深度（米）
  stair_width: 1.20     # 楼梯宽度（米）
  window: {wall: north, pos_z: 2.10, ...}   # 楼梯上方高窗
  lighting_preset: soft_ambient
```

#### 1.2 新物体（7 个）

**文件**：`asset_library/objects.yaml`

| 名称 | 形状 | 类别 | 说明 |
|------|------|------|------|
| `rubber_ball_st` | sphere r=0.08m | safe | 软橡皮球，楼梯上随机弹跳 |
| `wooden_block_st` | box 0.08³m | safe | 木块，翻滚下楼 |
| `bowling_ball_st` | sphere r=0.108m | dangerous | 保龄球，重且快 |
| `ceramic_vase_st` | cylinder | dangerous | 陶瓷花瓶，破碎危险 |
| `metal_canister_st` | cylinder | dangerous | 金属罐，高密度 |
| `foam_cube_st` | box 0.09³m | adversarial | 看起来沉，实为泡沫 |
| `steel_ball_st` | sphere r=0.08m | adversarial | 看起来像橡皮球，实为钢球 |

#### 1.3 SceneSpec 扩展

**文件**：`pipeline/scene_spec.py`

新增 `StairTumbleSpec` 数据类：

```python
class StairTumbleSpec(BaseModel):
    n_steps: int        # 楼梯级数
    step_rise: float    # 踏步高度
    step_run: float     # 踏步深度
    stair_width: float
    stair_x: float      # 楼梯中心 X
    stair_start_y: float  # = depth/2 - n_steps×step_run
    start_step: int     # 物体出生台阶（从 0 计）
    start_x: float
    nudge_vel_y: float  # 初始向南推速（< 0）
    euler_z: float
```

`SceneSpec` 新增 `stair: Optional[StairTumbleSpec]` 字段，并在 `object_world_pos()` 和 `validate_geometry()` 中支持该任务类型。

#### 1.4 Randomizer 扩展

**文件**：`pipeline/randomizer.py`

新增方法：

| 方法 | 作用 |
|------|------|
| `_build_stair_tumble_spec()` | 组装完整 SceneSpec；GT: 重物→`BRACE_FOR_IMPACT`，轻物→`TRIGGER_DODGE` |
| `_make_stair()` | 采样 StairTumbleSpec（起始台阶、推速、X 偏移） |
| `_stair_world_pos()` | 计算物体 t=0 世界坐标（供相机生成用） |
| `_make_stair_cameras()` | 生成 3 个相机：observer/closeup/overhead |

关键几何公式：
```python
stair_start_y = room_depth / 2 - n_steps * step_run   # 楼梯底部 Y
obj_y = stair_start_y + start_step * step_run + step_run / 2
obj_z = (start_step + 1) * step_rise + hz
```

`Randomizer.__init__` 中：当 `task_type == "stair_tumble"` 时，`room_types` 默认仅含 `{"stair_landing"}`。

#### 1.5 SceneBuilder 扩展

**文件**：`scene_builder.py`

新增方法：

| 方法 | 作用 |
|------|------|
| `_add_stair_geometry()` | 用 N 个 Box 实体拼出楼梯（累积高度法，碰撞正确） |
| `_add_stair_tumble_object()` | 将物体放置在指定台阶踏面上 |
| `_apply_stair_velocity()` | 施加初始向南推速 `nudge_vel_y` |

楼梯几何：台阶 k 的 Box 高度为 `(k+1) × step_rise`（累积实心，下方无空洞），
位置 `pos_y = stair_start_y + k×step_run + step_run/2`。

---

### 2. LLM 模式：物理 Hints 系统

**背景**：LLM 模式原来只返回 `(task_type, object_name, room_type)`，无法从描述中提取速度/高度偏好。新增可选的 `hints` 字段，让描述文字影响物理采样范围。

#### 2.1 Randomizer 侧

**文件**：`pipeline/randomizer.py`

```python
# 新增实例变量
self._active_hints: dict = {}

# sample_with_constraints 新增参数
def sample_with_constraints(self, seed, force_object=None,
                             force_room=None, hints=None) -> SceneSpec:
    self._active_hints = hints or {}
    ...
    self._active_hints = {}   # 调用后重置
```

新增两个 helper：

```python
def _hint_range(self, lo, hi, hi_max=1e9) -> tuple[float, float]:
    """按 speed hint 缩放采样区间。负值区间同样正确处理。"""
    scale = {"slow": 0.4, "normal": 1.0, "fast": 1.8, "very_fast": 3.0}[hint]
    new_hi = min(hi * scale, hi_max)
    new_lo = min(lo * scale, new_hi * 0.95)   # 防止 lo > hi（角度上限夹住时）
    return new_lo, new_hi

def _hint_height(self, lo, hi) -> tuple[float, float]:
    """按 height hint 平移采样区间。"""
    span = hi - lo
    if height == "low":  return max(0.1, lo - span*0.3), lo
    if height == "high": return hi, hi + span*0.3
    return lo, hi
```

14 个 `_make_*` 方法均接入 `_hint_range` / `_hint_height`：

| 任务 | 受 speed 影响的参数 | 受 height 影响的参数 |
|------|---------------------|----------------------|
| object_drop | `vel_x` | — |
| furniture_tip | `angular_vel` | — |
| hanging_fall | `angular_vel` | `attach_z` |
| sliding_object | `angle_deg`（上限 70°） | — |
| stack_collapse | `angular_vel` | — |
| rolling_ball | `vel_x` | — |
| shelf_slide | `vel_y` | `height` |
| door_swing | `angular_vel` | — |
| thrown_object | `vel_y` | — |
| pendulum_swing | `initial_angle_deg`（上限 85°） | — |
| bouncing_object | `drop_h`, `vel_y` | — |
| ladder_slip | `angular_vel` | — |
| chain_reaction | `trigger_vel_y` | — |
| stair_tumble | `nudge_vel_y` | `start_step` |

#### 2.2 LLMPlanner 侧

**文件**：`pipeline/llm_planner.py`

- `_SYSTEM_PROMPT` 新增 **Hints** 段落，说明 `speed`/`height` 字段含义、取值和触发短语
- `_llm_select()` 返回值从 3-tuple 改为 4-tuple：`(task_type, obj_name, room_type, hints)`
- `plan()` / `plan_with_variants()` 将 hints 传给 `sample_with_constraints(hints=…)`
- JSON 输出示例：
  ```json
  {"task_type": "stair_tumble", "object_name": "rubber_ball_st",
   "room_type": "stair_landing", "hints": {"speed": "fast"}}
  ```

验证效果（seed=0）：
```
nudge_vel_y  default : -0.347 m/s
nudge_vel_y  fast    : -0.624 m/s   (~1.8×)
```

---

### 3. LLM 模式：`stair_tumble` 任务接入

**文件**：`pipeline/llm_planner.py`

- `_TASK_TYPES` 列表新增 `"stair_tumble"`（共 15 种任务）
- `_load_catalogues()` 新增 `self._stair_obj_names`，读取 `task_type == "stair_tumble"` 的物体
- `_task_catalogue` 映射新增 `"stair_tumble": self._stair_obj_names`
- `_build_user_message()` 新增 `Stair Tumble Catalogue` 参数
- `_SYSTEM_PROMPT` 新增 `stair_tumble` 任务描述及 Rule 2 中的目录映射

**文件**：`docs/llm_mode.md`

- 任务类型表：14 → 15 种
- 房间目录：4 → 5 种（新增 `stair_landing`）
- 新增完整的 **Physics Hints** 章节（speed/height 表、参数映射、代码示例）

---

### 4. LLM 物体匹配策略：优先相似匹配

**文件**：`pipeline/llm_planner.py`

原 Rule 1：仅当名称完全一致时才使用目录中已有物体，否则触发 Objaverse 下载。

**问题**：描述 "a football rolling down the stair" 中 "football" 不在目录，
LLM 触发下载并选择了无效的 LVIS label `"football"`，导致 Objaverse 查找失败。

**新 Rule 1**：优先按物理类型和功能匹配目录中最接近的物体：
- "football" / "soccer ball" → 最近的球形物体（如 `rubber_ball_st`）
- 仅当目录中没有任何形状/行为相似的物体时，才触发 Objaverse 下载

同时在 `_SYSTEM_PROMPT` 末尾补充常见 LVIS label 对照表：
```
"football" → "soccer_ball" 或 "football_(American)"
"knife"    → "kitchen_knife"
"cup"      → "mug"
```

---

### 5. Bug 修复

#### 5.1 trimesh 4.x API 变更

**文件**：`pipeline/llm_planner.py`

```python
# 修复前（trimesh < 4.x 模块级函数，4.x 已移除）
mesh = trimesh.simplify_quadric_decimation(mesh, face_count=8000)

# 修复后（实例方法）
mesh = mesh.simplify_quadric_decimation(face_count=8000)
```

#### 5.2 Objaverse Mesh 不封闭导致渲染透明

**背景**：从 Objaverse 下载的 soccer ball GLB 由多个六边形/五边形面片拼成，
面片之间有缝隙（`is_watertight=False`），Genesis 渲染时从缝隙穿透，球体看起来透明。

**修复**：注册 `football_st` 时，用凸包（convex hull）替换原始合并 mesh：

```python
hull = mesh.convex_hull      # 保证封闭、无缝隙
hull = normalize(hull, 0.22)
hull.export("football_st.obj", ...)
```

| | 原始 mesh | 修复后（convex hull） |
|--|-----------|----------------------|
| 面数 | 92,160 → 46,882（简化不足） | 25,732 |
| `is_watertight` | False | True |
| Genesis 渲染 | 透明（缝隙穿透） | 实体白色球 |

---

### 改动文件汇总

| 文件 | 改动类型 |
|------|---------|
| `asset_library/rooms.yaml` | 新增 `stair_landing` 房间 |
| `asset_library/objects.yaml` | 新增 7 个 `stair_tumble` 物体；自动注册 `football_st`（凸包 mesh） |
| `pipeline/scene_spec.py` | 新增 `StairTumbleSpec`；`SceneSpec` 新增 `stair` 字段；`object_world_pos` / `validate_geometry` 支持 stair_tumble |
| `pipeline/randomizer.py` | 新增 `_hint_range` / `_hint_height`；14 个 `_make_*` 接入 hints；新增 stair_tumble 全套方法；`sample_with_constraints` 加 `hints` 参数 |
| `pipeline/llm_planner.py` | `_TASK_TYPES` +1；stair 目录接入；hints 系统（解析+传递）；Rule 1 改为相似匹配；trimesh API 修复；LVIS label 对照表 |
| `scene_builder.py` | 新增 `_add_stair_geometry` / `_add_stair_tumble_object` / `_apply_stair_velocity` |
| `docs/llm_mode.md` | 任务/房间表更新；新增 Physics Hints 完整章节 |
| `assets/meshes/football_st.obj` | Objaverse soccer ball 凸包 mesh（25,732 面，封闭） |

## Session: 2026-04-16

---

### 1. 环境修复

| 问题 | 修复 |
|------|------|
| 缺少 EGL 库，headless 渲染报 `NoneType` 错误 | `sudo apt-get install libegl-mesa0 libegl1` |
| `pydantic` 未安装 | `pip install pydantic` |
| `genesis-world` 未安装 | `pip install genesis-world` |
| PyTorch 版本过旧（`< 2.8.0`） | `pip install --upgrade torch` |
| NumPy 2.x 与旧编译扩展不兼容 | `pip install "numpy<2"` |

---

### 2. Mesh 路径修复

**文件**: `scene_builder.py`

`MESHES_DIR` 指向了错误的父目录：

```python
# 修复前（错误）
MESHES_DIR = REPO_ROOT / "assets" / "meshes"
# → /lambda/nfs/Yizhan3d/assets/meshes/  （目录不存在）

# 修复后（正确）
MESHES_DIR = PROJECT_ROOT / "assets" / "meshes"
# → /lambda/nfs/Yizhan3d/genesis_scene_generation/assets/meshes/
```

---

### 3. Mesh 简化

**文件**: `assets/meshes/`

`foam_anvil.obj` 从 Objaverse 下载后有 278,528 个面（30 MB），Genesis 加载时挂死。
用 `trimesh.simplify_quadric_decimation(face_count=N)` 简化：

| 文件 | 简化前 | 简化后 |
|------|--------|--------|
| `foam_anvil.obj` | 278,528 面 / 30 MB | 6,674 面 / 223 KB |
| `coffee_mug.obj` | 14,080 面 / 1.4 MB | 4,000 面 / 144 KB |
| `hot_iron.obj` | 8,610 面 / 1.1 MB | 4,000 面 / 168 KB |
| `plastic_bottle.obj` | 9,408 面 / 970 KB | 4,000 面 / 136 KB |

---

### 4. Mesh 物理尺寸字段

**背景**：Mesh 类型物体的 `size` 字段只存储 scale 系数（`[1.0]`），不是物理半径。
原来 `_make_drop` 用 `obj.size[0]` 计算桌面 overhang 时，把 scale=1.0 当作 1 米的半径，
导致物体出生点超出桌面 0.5+ 米，悬在空中而非桌上。

**修复**：在 `ObjectSpec` 加两个字段：

```python
# pipeline/scene_spec.py
class ObjectSpec(BaseModel):
    ...
    phys_half_x: float = 0.0   # 物理半宽（metres）；0 = 由 size[] 自动推断
    phys_half_z: float = 0.0   # 物理半高（metres）；0 = 由 size[] 自动推断
```

`asset_library/objects.yaml` 中所有 mesh 物体补充实测值：

| 物体 | phys_half_x | phys_half_z |
|------|-------------|-------------|
| red_apple / steel_apple | 0.037 | 0.037 |
| coffee_mug | 0.055 | 0.043 |
| plastic_bottle | 0.037 | 0.037 |
| hot_iron | 0.120 | 0.120 |
| foam_anvil | 0.080 | 0.034 |

`pipeline/randomizer.py` 的 `_make_object_spec()` 补充读取这两个字段（之前遗漏，导致 yaml 里的值不生效）：

```python
def _make_object_spec(self, d: dict) -> ObjectSpec:
    return ObjectSpec(
        ...
        phys_half_x=d.get("phys_half_x", 0.0),   # ← 新增
        phys_half_z=d.get("phys_half_z", 0.0),   # ← 新增
    )
```

`_make_drop()` 使用正确的物理半径：

```python
# 修复前
obj_rx = obj.size[0]   # mesh 时 = 1.0（scale），错误

# 修复后
if obj.morph == "mesh" and obj.phys_half_x > 0:
    obj_rx = obj.phys_half_x   # 真实物理尺寸
else:
    obj_rx = obj.size[0]       # primitive 时 = 半径，正确
```

`scene_builder.py` 的 hanging_fall 落地检测也改用 `phys_half_z`：

```python
o = self.spec.object
if o.phys_half_z > 0:
    half_h = o.phys_half_z
elif len(o.size) >= 3:
    half_h = o.size[2]
else:
    half_h = o.size[0]
```

---

### 5. Hanging Fall 相机视角修复

**背景**：`hanging_fall` 场景的相机 lookat 硬编码为 `y=0.0`（房间中心），
但 `wall_north` 类型的物体挂在 `y ≈ +depth/2`（北墙）。
相机完全朝向相反方向，拍出空白视频。

**根本原因**：`_make_hanging_cameras()` 里用 `hanging.pos_y`（永远是 0.0）
作为 lookat Y，而物体实际 Y 坐标是在 `scene_builder._add_hanging_object()` 里
才计算的（`wall_y = room.depth / 2`），randomizer 侧不知道这个值。

**修复**：在 `Randomizer` 加几何辅助方法，把物体世界坐标的计算集中到 randomizer 侧，
相机生成直接用真实坐标：

```python
# pipeline/randomizer.py — 新增静态方法

@staticmethod
def _obj_phys_half(obj):
    """对所有 morph 类型返回一致的 (half_x, half_y, half_z)"""
    if obj.morph == "mesh":
        hx = obj.phys_half_x if obj.phys_half_x > 0 else 0.05
        hz = obj.phys_half_z if obj.phys_half_z > 0 else 0.05
        return hx, hx, hz
    elif obj.morph == "sphere": ...
    elif obj.morph == "cylinder": ...
    else:  # box
        return obj.size[0], obj.size[1], obj.size[2]

@staticmethod
def _drop_world_pos(table, drop, obj):
    """object_drop 物体 t=0 的世界坐标"""
    _, _, hz = Randomizer._obj_phys_half(obj)
    obj_z = table.height - obj.bottom_z_offset + hz  # mesh
    # 或 table.height + hz  # primitive
    return drop.start_x, drop.start_y, obj_z

@staticmethod
def _hanging_world_pos(room, hanging, obj):
    """hanging_fall 物体 t=0 的世界坐标"""
    if hanging.attachment == "wall_north":
        _, hy, _ = Randomizer._obj_phys_half(obj)
        wall_y = room["depth"] / 2
        return hanging.pos_x, wall_y - hy, hanging.attach_z
    else:  # ceiling
        return hanging.pos_x, hanging.pos_y, hanging.attach_z

@staticmethod
def _tip_world_pos(tip, obj):
    """furniture_tip 物体 t=0 的世界坐标"""
    _, _, hz = Randomizer._obj_phys_half(obj)
    return tip.start_x, tip.start_y, hz
```

各 `_make_*_cameras()` 函数改为接收并使用 `obj_world_pos`，不再 hardcode 偏移。

---

### 6. 几何验证器（`validate_geometry`）

**文件**: `pipeline/scene_spec.py`

在 `SceneSpec` 新增两个方法：

**`object_world_pos()`** — 从 spec 字段反算物体世界坐标，三种任务类型均支持：

```python
def object_world_pos(self) -> tuple[float, float, float]:
    """Return the world-space centre of the object at t=0."""
    ...
```

**`validate_geometry()`** — 轻量级几何检查，返回警告列表（空列表 = 通过）：

```python
def validate_geometry(self) -> list[str]:
    """检查项：
    1. 物体在房间边界内（wall_north 物体豁免边界检查）
    2. 物体在地板之上
    3. 每个相机的 lookat 方向与物体方向夹角 < FOV × 85%
    """
```

**`batch_runner.py`** 在启动仿真前调用验证，快速失败：

```python
geo_warnings = spec_obj.validate_geometry()
if geo_warnings:
    return {"success": False, "stderr": "GEOMETRY: " + "; ".join(geo_warnings)}
```

**效果**：把原来 "等 180s 超时才知道失败" 变成 "不到 1s 立即报告几何错误"。

---

### 7. 相机生成重试机制

**文件**: `pipeline/randomizer.py`

新增 `_make_cameras_with_retry()`：

```python
def _make_cameras_with_retry(self, rng, make_fn, obj_pos, max_tries=6):
    """调用 make_fn 最多 max_tries 次，返回第一个通过几何检查的结果。"""
    for _ in range(max_tries):
        cameras = make_fn(rng)
        if self._cameras_pass(cameras, obj_pos):
            return cameras
    return cameras  # 返回最后一次结果
```

三个 `_build_*_spec()` 方法全部改用此 wrapper：

```python
obj_pos = self._drop_world_pos(table, drop_spec, obj_spec)
cameras = self._make_cameras_with_retry(
    rng,
    lambda r: self._make_cameras(r, table, drop_spec, obj_spec),
    obj_pos,
)
```

**最终验证结果（每种任务 200 个 spec）**：

| 任务类型 | 几何警告率（修复前） | 几何警告率（修复后） |
|----------|---------------------|---------------------|
| object_drop | ~11% | 0% |
| furniture_tip | ~1% | 0% |
| hanging_fall | ~60% | 0% |

---

### 改动文件汇总

| 文件 | 改动类型 |
|------|---------|
| `scene_builder.py` | 修复 `MESHES_DIR` 路径；hanging_fall 落地检测用 `phys_half_z` |
| `pipeline/scene_spec.py` | `ObjectSpec` 加 `phys_half_x/z`；`SceneSpec` 加 `object_world_pos()` + `validate_geometry()` |
| `pipeline/randomizer.py` | 加 `_obj_phys_half` / `_drop_world_pos` / `_hanging_world_pos` / `_tip_world_pos`；修复所有 camera lookat；加重试机制；`_make_object_spec` 读取 `phys_half_x/z` |
| `pipeline/batch_runner.py` | 仿真前调用 `validate_geometry()` 快速失败 |
| `asset_library/objects.yaml` | 6 个 mesh 物体补充 `phys_half_x/z` |
| `assets/meshes/*.obj` | 4 个过大 mesh 简化到 4000–6700 面 |
