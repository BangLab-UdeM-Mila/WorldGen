# ReactHuman — Pipeline 现状与下一步计划

> 更新日期：2026-04-17

---

## 一、整体架构

```
描述 / 随机种子
      │
      ▼
┌─────────────────────────────────────────────┐
│              SceneSpec (Pydantic)            │
│  task_type · room · object · drop/tip/      │
│  hanging/stack/ramp · cameras · lighting    │
└─────────────────────────────────────────────┘
      │                          │
      ▼                          ▼
 Randomizer               LLMPlanner
 (procedural)             (Claude API)
      │                          │
      └──────────┬───────────────┘
                 ▼
          SceneSpec JSON
                 │
                 ▼
          validate_geometry()        ← 几何预检，无需仿真
                 │
                 ▼
          SceneBuilder               ← Genesis GPU 仿真
          (scene_builder.py)
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
     视频帧    元数据    ground-truth
    (MP4)    (JSON)     labels
```

---

## 二、当前已完成（Phase 1）

### 2.1 已实现的 5 个任务类型

| 任务类型 | 说明 | 物体数 | 状态 |
|----------|------|--------|------|
| `object_drop` | 小物体从桌面重心越界自由落下 | 10 | ✅ 完成并测试 |
| `sliding_object` | 物体沿斜面滑向观察者 | 共用 drop 库 | ✅ 完成并测试 |
| `stack_collapse` | 叠放物体整体向观察者方向倒塌 | 4 | ✅ 完成并测试 |
| `hanging_fall` | 墙面/天花板挂件脱落 | 7 | ✅ 完成并测试 |
| `furniture_tip` | 高耸家具向观察者倾倒 | 7 | ✅ 完成并测试 |

**Asset library 合计：28 个物体**（含对抗样本 4 个）

### 2.2 两种生成模式

#### Procedural 模式（主力，无需 API）

```
python generate_dataset.py \
    --task-type object_drop \
    --n 1000 \
    --adversarial-prob 0.15 \
    --workers 4 \
    --output dataset/
```

数据流：
1. `Randomizer.sample_batch(n, seed_start)` → 随机采样 SceneSpec
2. 每个 spec 调用 `validate_geometry()` 预检（几何越界、相机视角）
3. `BatchRunner` 多进程调度 → 每个 spec 独立子进程运行 Genesis
4. `SceneBuilder` 构建场景 → 仿真 → 写出 `{scene_id}/video.mp4` + `metadata.json`

随机化维度：
- 物体选择（类别过滤 + 对抗概率）
- 房间类型 + 光照预设
- 放置位置、初始速度/角速度、旋转角
- 相机位置（多角度 retry，确保物体在 FOV 内）

#### LLM 模式（创意多样，需 ANTHROPIC_API_KEY）

```
python generate_dataset.py \
    --mode llm \
    --descriptions scenarios.txt \
    --variants 20 \
    --output dataset/
```

数据流：
1. 读取自然语言描述文件（每行一个场景描述）
2. `LLMPlanner._llm_select(description)` → 单次 Claude API 调用
   - 返回：`task_type` · `object_name` · `room_type`
   - 若 LLM 提议新物体（`new_object` 字段）：
     - 从 Objaverse 下载 GLB → trimesh 归一化 + 面数简化（≤8,000 面）
     - 导出 `.obj` → 追加注册到 `objects.yaml`
3. `Randomizer.sample_with_constraints(seed, force_object, force_room)` × N 变体
4. 后续与 procedural 模式相同

LLM 已注册的任务类型（5 个 catalogue 均已对齐）：

| task_type | Catalogue 物体数 | 说明 |
|-----------|-----------------|------|
| `object_drop` | 10 | 桌面可接/可闪物体 |
| `sliding_object` | 共用 drop | 同一物体库 |
| `furniture_tip` | 7 | 高家具 |
| `hanging_fall` | 7 | 挂墙/天花板件 |
| `stack_collapse` | 4 | 可叠放件 |

### 2.3 关键设计不变量

| 不变量 | 保障机制 |
|--------|---------|
| `_xxx_world_pos()` 公式必须与 `scene_builder` 放置完全一致 | `validate_geometry()` 在仿真前检测 |
| 相机必须看到物体（角度 < FOV × 85%） | 相机生成带 retry（最多 6 次）|
| mesh 物体必须填写 `phys_half_x/z` | `validate_assets.py` 静态检查 |
| mesh 面数 ≤ 15,000 | `validate_assets.py` 静态检查 |
| SceneSpec 相同 seed → 仿真 100% 可复现 | Genesis 固定 dt/substeps/seed |

---

## 三、下一步计划

### Phase 2 — 新任务类型（难度：中）

目标：完成 BENCHMARK_DESIGN.md 中 Phase 2 的 3 个任务类型。

#### 3.1 `rolling_ball`（A3）

- **物理机制**：球有初速度，在台面滚动后越过桌边抛体落下。  
- **新增 spec**：`RollSpec`（初速度、初始位置、自旋角速度）  
- **场景构建**：球必须使用 `gs.morphs.Sphere`，摩擦力驱动滚动而非滑动  
- **新增物体**：橡皮球、网球、台球、铅芯网球（对抗）  
- **关键难点**：滚动 vs. 静止摩擦系数需要精细调参，否则球会"滑"而不"滚"

#### 3.2 `shelf_slide`（A5）

- **物理机制**：壁架上的物体因微小振动越过架边滑落（类似 `object_drop` 但起点在高处壁架而非桌面）  
- **新增 spec**：`ShelfSpec`（架子高度、壁挂位置、架深）  
- **场景构建**：需要在墙面生成壁架几何体（盒形 + 两个支撑臂）  
- **新增物体**：花盆、摆件、调料瓶、泡沫花瓶（对抗）  
- **关键难点**：壁架几何体需要固定在墙壁上但不影响物体碰撞

#### 3.3 `door_swing`（C2）

- **物理机制**：门绕合页轴旋转，用旋转关节约束实现  
- **新增 spec**：`DoorSpec`（门宽/高/厚、初始角度、初始角速度、门材质）  
- **场景构建**：Genesis 的 `gs.morphs.Box` + revolute joint  
- **新增物体**：实木门、玻璃门、空心门（对抗）  
- **关键难点**：Genesis 旋转关节 API，门落在 `wall_south` 或 `wall_east` 合适位置

**Phase 2 预计工期**：每个任务类型约 1–2 天（spec → randomizer → scene_builder → 测试）

---

### Phase 3 — 约束体与复合动力学（难度：高）

| 任务 | 新增技术需求 |
|------|------------|
| `thrown_object`（D1） | 抛体初速度（方向 + 大小）、飞行轨迹预测用于 L4 评估 |
| `pendulum_swing`（E1） | 绳约束（Genesis `constraint` API）、摆动轨迹计算 |
| `curtain_rod_fall`（B3） | 布料 + 刚体混合（窗帘布料 mesh + 杆刚体） |

---

### Phase 4 — 评估基础设施

与任务类型扩展并行推进，这些是 benchmark 可用的前提。

#### 4.1 Ground-truth 后处理（当前：字段存在但未填充）

`SceneSpec` 已有字段：
```python
interception_point_2d: Optional[list[float]]  # [px_x, px_y]
interception_point_3d: Optional[list[float]]  # [x, y, z]
time_to_floor_s:       Optional[float]
```

需要在 `SceneBuilder._build_metadata()` 中实际计算：
- `time_to_floor_s`：从仿真轨迹中找第一次 `obj.pos[2] < floor_threshold` 的时间戳  
- `interception_point_3d`：对象到达 `z = hip_height`（~0.9 m）时的 world 坐标  
- `interception_point_2d`：将 3D 点投影到每个相机的像素坐标  

#### 4.2 数据集打包与统计

- 生成 `dataset/manifest.json`：所有 scene 的 id、task_type、gt_action、adversarial 汇总  
- 生成 `dataset/stats.md`：各类别分布、对抗比例、平均 time_to_floor 等  
- 验证脚本：检查每个 scene 目录结构完整（video + metadata + spec）

#### 4.3 评估脚本（Track 1）

针对模型输出 `EXECUTE_CATCH / TRIGGER_DODGE / BRACE_FOR_IMPACT` 的分类准确率：
- 整体 Accuracy  
- 按 task_type 分层 Accuracy  
- 对抗样本 `adv_correct` / `adv_fooled` 比率  
- `adv_switch_frame`：从错误动作切换到正确动作的帧数

---

### Phase 5 — 长期（Genesis 流体 / 布料）

| 任务 | 依赖 |
|------|------|
| `liquid_spill` | Genesis MPM 流体仿真稳定性 |
| `ceiling_tile_fall`（B4） | 多刚体碎裂（fractured mesh） |
| `chain_reaction`（F1） | 触发器逻辑 + 多物体轨迹追踪 |

---

## 四、下一步行动（优先级排序）

| 优先级 | 任务 | 预计工时 |
|--------|------|---------|
| 🔴 P0 | `interception_point_3d/2d` 和 `time_to_floor_s` 后处理实现 | 0.5 天 |
| 🔴 P0 | `rolling_ball` 任务类型实现 | 1.5 天 |
| 🟠 P1 | `shelf_slide` 任务类型实现 | 1 天 |
| 🟠 P1 | 数据集 manifest + stats 打包脚本 | 0.5 天 |
| 🟡 P2 | `door_swing` 任务类型（旋转关节） | 2 天 |
| 🟡 P2 | Track 1 评估脚本 | 1 天 |
| 🟢 P3 | Phase 3 任务类型（thrown / pendulum / curtain） | 各 2–3 天 |
