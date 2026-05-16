# Quickstart

这份文档是快速上手指南和对接方案。
`docs/DESIGN.md`,完整说明见 `README.md`。

## 这个模块在干什么

把模型的预测和 simulator 的标准答案对一下,算出一份评测报告。
分两部分:

- **Track 1(语义)**:模型选对动作了吗?(接 / 躲 / 撑)
- **Track 2(运动学)**:模型预测的接住位置准不准?

加上一个**难度切片**:每个场景有三个 0-1 之间的难度分(initial_state /
action / physics),最后能告诉你"模型在哪一个维度最弱"。

## 输入是什么

两样东西:

1. **数据集目录**(Yizhan 生成的),里面每个场景有自己的文件夹:

```
dataset_furniture_test/
  bc51dcdc/
    spec.json          ← 标准答案在这里
    metadata.json
    video_observer.mp4
    frame_start.png
    frame_mid.png
    frame_end.png
  ec91e8c0/
    ...
```

2. **模型预测文件**(`.jsonl`),每行一个 JSON,长这样:

```jsonl
{"scene_id": "bc51dcdc", "action": "EXECUTE_CATCH", "target_point_world": [0.11, 0, -0.33], "latency_seconds": 1.8, "rationale": "..."}
{"scene_id": "ec91e8c0", "action": "TRIGGER_DODGE", "latency_seconds": 1.4, "rationale": "..."}
```

## 怎么跑

```bash
# 安装
pip install -e .

# 跑一个内置 demo,确认环境没问题
python examples/run_demo.py

# 跑真实数据
reacthuman \
  --dataset path/to/dataset_furniture_test \
  --predictions path/to/claude_predictions.jsonl \
  --model-name claude-opus-4.7 \
  --out report_claude.json
```

## 输出是什么

一个 JSON 文件,里面有:

```json
{
  "model_name": "claude-opus-4.7",
  "n_scenes": 500,
  "semantic_action_accuracy": 0.73,
  "fatal_execution_rate": 0.08,
  "adversarial_fool_rate": 0.42,
  "trajectory_error_cm_mean": 14.2,
  "trajectory_error_cm_median": 11.8,
  "reachable_prediction_rate": 0.61,
  "by_initial_state": { "easy": 0.91, "medium": 0.78, "hard": 0.52, "adversarial": 0.31 },
  "by_action": { "easy": 0.85, "medium": 0.72, "hard": 0.55, "adversarial": 0.40 },
  "by_physics": { "easy": 0.92, "medium": 0.75, "hard": 0.48, "adversarial": 0.18 },
  "by_task_family": { "object_drop": 0.81, "sliding_object": 0.70, ... }
}
```

## 需要跟 Yizhan 对接的两件事

1. **确认 `spec.json` 的字段格式跟 `reacthuman/io.py` 里写的一致**。
   如果字段名不一样(比如yizhan用 `gt_action` 而不是 `correct_action`),
   只需要改 `io.py::load_scene_spec`,其他代码都不用动。

2. **请yizhan在 `spec.json` 里加四个字段**(如果还没有):
   - `difficulty.initial_state`
   - `difficulty.action`
   - `difficulty.physics`
   - 对于 adversarial 场景,在 `extras.appearance_implied_action`
     里写上"如果模型只看外观会选哪个动作"

   这两步做完,整条评测链就接通了。

## 跑测试

```bash
pytest -q
```

应该看到 `14 passed`。如果有失败,先看是不是 Python 版本太低
(需要 ≥ 3.9)。
