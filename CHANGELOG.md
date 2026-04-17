# ReactHuman Pipeline — Changelog

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
