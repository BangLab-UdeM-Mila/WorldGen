# ReactHuman — Pipeline 现状与下一步计划

> 更新日期：2026-04-22

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
| `rolling_ball` | 球从桌面滚落，滚动而非滑动 | 6 | ✅ 完成，几何验证通过（200 specs） |
| `shelf_slide` | 壁架上物体滑落（高位坠落） | 5 | ✅ 完成，几何验证通过（200 specs） |
| `door_swing` | 门绕合页轴失控旋转 | 5 | ✅ 完成，有测试数据（dataset_door_test） |
| `thrown_object` | 物体被人为抛出飞向观察者 | 6 | ✅ 完成，有测试数据（dataset_thrown_test） |
| `pendulum_swing` | 重物做钟摆运动 | 5 | ✅ 完成，有测试数据（dataset_pendulum_test） |
| `bouncing_object` | 球落地弹跳向观察者 | 6 | ✅ 完成，仿真测试通过（10 scenes，25s/场景） |

**Asset library 合计：57 个物体**（含对抗样本 14 个）

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

#### 3.1 `rolling_ball`（A3）✅ 已完成

- **实现方式**：`RollSpec`（start_x/y, vel_x/y, angular_vel_y）  
- **rolling-without-slipping**：ω_y = vel_x / radius 在 Randomizer 中预计算并存入 spec  
- **新增物体**：rubber_ball, tennis_ball, billiard_ball, bowling_ball, foam_billiard, lead_tennis  
- **几何验证**：200 specs 全部通过

#### 3.2 `shelf_slide`（A5）✅ 已完成

- **实现方式**：`ShelfSpec`（height, pos_x, depth, width, thickness, vel_y）  
- **场景构建**：主板 + 两个支撑臂，全部 fixed=True；物体给初速度 vel_y < 0 触发滑落  
- **新增物体**：small_plant, spice_bottle, ceramic_vase, heavy_toolbox, foam_vase  
- **几何验证**：200 specs 全部通过（closeup 相机 lookat-z 已修复）

#### 3.3 `door_swing`（C2）✅ 已完成

- **实现方式**：`DoorSpec`（门宽/高/厚、初始角度、初始角速度）  
- **场景构建**：Genesis revolute joint + 南墙开门洞  
- **新增物体**：wooden_door, hollow_door, glass_door, steel_door, foam_door  
- **几何验证**：通过（dataset_door_test 有测试数据）

---

### Phase 3 — 约束体与复合动力学

| 任务 | 新增技术需求 | 状态 |
|------|------------|------|
| `thrown_object`（D1） | 抛体初速度（方向 + 大小） | ✅ 完成（dataset_thrown_test） |
| `pendulum_swing`（E1） | MJCF hinge joint 约束 | ✅ 完成（dataset_pendulum_test） |
| `curtain_rod_fall`（B3） | 布料 + 刚体混合 | 🔲 待实现（需 Genesis cloth） |

---

### Phase 4 — Phase 4 任务类型（难度：中）

| 任务 | 新增技术需求 | 状态 |
|------|------------|------|
| `bouncing_object`（D2） | 后弹跳弹道（post-bounce 参数化） | ✅ 完成，10 scenes 仿真测试通过，25s/场景 |
| `ladder_slip`（C3） | 梯子在光滑地板上侧滑倒塌 | 🔲 待实现 |
| `chain_reaction`（F1） | 触发器逻辑 + 多物体轨迹 | 🔲 待实现 |

---

### Phase 5 — 评估基础设施

#### 5.1 Ground-truth 后处理 ✅ 基础实现完成

`SceneBuilder._build_metadata()` 已计算：
- `time_to_floor_s`：仿真中记录首次落地/事件帧  
- `interception_point_3d`：对象到达 `z = hip_height`（~0.9 m）时的 world 坐标  
- `interception_point_2d`：投影到各相机像素（部分场景仍为 null）

#### 5.2 数据集打包与统计

- `validate_scene.py` ✅：场景质量验证（--json 输出汇总）  
- manifest.json + stats.md：🔲 待实现（独立脚本）

#### 5.3 评估脚本（Track 1）🔲 待实现

针对模型输出 `EXECUTE_CATCH / TRIGGER_DODGE / BRACE_FOR_IMPACT` 的分类准确率：
- 整体 Accuracy  
- 按 task_type 分层 Accuracy  
- 对抗样本 `adv_correct` / `adv_fooled` 比率  
- `adv_switch_frame`：从错误动作切换到正确动作的帧数

---

### Phase 6 — 长期（Genesis 流体 / 布料）

| 任务 | 依赖 |
|------|------|
| `liquid_spill` | Genesis MPM 流体仿真稳定性 |
| `ceiling_tile_fall`（B4） | 多刚体碎裂（fractured mesh） |
| `curtain_rod_fall`（B3） | Genesis cloth + 刚体混合 |

---

## 四、下一步行动（优先级排序）

| 优先级 | 任务 | 预计工时 | 状态 |
|--------|------|---------|------|
| 🔴 P0 | `interception_point_3d/2d` 和 `time_to_floor_s` 后处理实现 | 0.5 天 | ✅ 完成 |
| 🔴 P0 | `rolling_ball` 任务类型实现 | 1.5 天 | ✅ 完成 |
| 🟠 P1 | `shelf_slide` 任务类型实现 | 1 天 | ✅ 完成 |
| 🟠 P1 | `door_swing` 任务类型（旋转关节） | 2 天 | ✅ 完成 |
| 🟠 P1 | `thrown_object` 任务类型 | 1.5 天 | ✅ 完成 |
| 🟠 P1 | `pendulum_swing` 任务类型（MJCF hinge） | 1.5 天 | ✅ 完成 |
| 🟠 P1 | `bouncing_object` 任务类型 | 1 天 | ✅ 完成，仿真通过（10 scenes 全 OK，25s/场景）|
| 🟡 P2 | 数据集 manifest + stats 打包脚本 | 0.5 天 | ✅ 完成（generate_manifest.py） |
| 🟡 P2 | Track 1 评估脚本 | 1 天 | ✅ 完成（evaluate_track1.py） |
| 🟡 P2 | `ladder_slip` 任务类型 | 1.5 天 | ✅ 完成（几何验证通过 50 specs） |
| 🟢 P3 | `chain_reaction` 任务类型 | 2–3 天 | ✅ 完成（几何验证通过 50 specs） |
| 🟢 P3 | `curtain_rod_fall` 任务类型（布料物理） | 3–4 天 | 待实现（需 Genesis cloth） |
