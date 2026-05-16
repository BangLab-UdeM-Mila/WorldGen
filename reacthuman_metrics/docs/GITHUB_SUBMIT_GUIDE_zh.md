# 把代码提交到 GitHub 你的 branch — 手把手教程

写给:Mengyang
目标:把 `reacthuman_metrics/` 这套代码提交到
`BangLab-UdeM-Mila/WorldGen` 的 `mengyang/physics-benchmark` 分支。
保密。

---

## 第 0 步:先理解发生了什么(不是命令,是概念)

GitHub 上的代码协作分四个"地方":

| 名字 | 在哪 | 是什么 |
|------|------|------|
| **远程仓库 (remote)** | GitHub 服务器上 | 大家共享的代码,网页上能看到的 |
| **本地仓库 (local repo)** | 你电脑的硬盘上 | 你自己电脑里的代码副本 |
| **工作区 (working tree)** | 你电脑的硬盘上 | 你正在编辑的文件 |
| **暂存区 (staging area)** | 介于工作区和本地仓库之间 | "我准备好要提交的这一波改动" |

提交一次代码 = 把改动 **从工作区 → 暂存区 → 本地仓库 → 远程仓库** 一路推上去。

下面每一步我都告诉你"这一步在做什么"。

---

## 第 1 步:确认电脑上有 Git,并配置好身份

打开终端(Mac 上是 Terminal,Windows 上是 Git Bash 或 PowerShell),输入:

```bash
git --version
```

如果显示版本号(例如 `git version 2.39.0`),说明已经装了。如果没有,
Mac 装一下 `brew install git`,Windows 去 https://git-scm.com/download 下载。

然后**只需要做一次**:告诉 Git 你是谁。这两行的邮箱要和你 GitHub
账号注册的邮箱一致:

```bash
git config --global user.name "Mengyang Xiong"
git config --global user.email "你的GitHub邮箱@example.com"
```

---

## 第 2 步:配置 SSH 密钥(强烈推荐,一次配好终生不用输密码)

GitHub 现在不接受网页密码登录,要用 **SSH key** 或 **personal access token**。
SSH 更稳定,推荐用 SSH。

### 2.1 生成密钥

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱@example.com"
```

按三次回车(默认路径、空密码、确认空密码)。

### 2.2 把公钥添加到 GitHub

```bash
# Mac:
cat ~/.ssh/id_ed25519.pub | pbcopy

# Linux:
cat ~/.ssh/id_ed25519.pub
# (然后手动复制输出的那一行)

# Windows (Git Bash):
cat ~/.ssh/id_ed25519.pub | clip
```

现在去浏览器:

1. 打开 https://github.com/settings/keys
2. 点右上角 **New SSH key**
3. Title 随便写(比如 "MacBook Pro Mengyang"),Key 那栏粘贴你刚复制的内容
4. 点 **Add SSH key**

### 2.3 测试连接

```bash
ssh -T git@github.com
```

如果出现 `Hi <你的用户名>! You've successfully authenticated...`,说明配好了。

---

## 第 3 步:把 WorldGen 仓库克隆到本地

选一个你想存代码的文件夹(比如 `~/Code`),然后:

```bash
cd ~/Code
git clone git@github.com:BangLab-UdeM-Mila/WorldGen.git
cd WorldGen
```

这一步做完,你的电脑上会有一个 `WorldGen` 文件夹,里面是整个仓库
的代码,而且自动连接到了 GitHub 的远程仓库。

---

## 第 4 步:切换到你的分支

```bash
# 先把所有远程分支信息拉下来
git fetch origin

# 看一下都有哪些分支
git branch -a
```

你应该能看到 `remotes/origin/mengyang/physics-benchmark` 在列表里。
切换过去:

```bash
git checkout mengyang/physics-benchmark
```

如果是第一次,Git 会自动创建本地分支并指向远程分支。看到
`Switched to branch 'mengyang/physics-benchmark'` 就对了。

确认一下当前在哪个分支:

```bash
git branch
# 应该显示:
#   Yizhan_scene_generation
# * mengyang/physics-benchmark   ← 带 * 号的是当前所在分支
#   main
```

---

## 第 5 步:把代码放进仓库

### 5.1 下载我给你的代码包

我把代码打包成了 `reacthuman_metrics.zip`(在 outputs 里)。
下载它,解压,你会得到一个 `reacthuman_metrics/` 文件夹。

### 5.2 决定放在哪

我建议放在 WorldGen 仓库的根目录下,作为一个独立模块:

```
WorldGen/
├── genesis_scene_generation/   ← Yizhan 的代码
├── reacthuman_metrics/         ← 你的代码,新加的
├── ...其他已经存在的目录...
```

具体操作:**把解压出来的整个 `reacthuman_metrics/` 文件夹复制到
WorldGen 仓库根目录下**。

```bash
# 假设解压后的代码在 ~/Downloads/reacthuman_metrics/
cp -r ~/Downloads/reacthuman_metrics ~/Code/WorldGen/
```

### 5.3 看一下 Git 检测到了什么

```bash
cd ~/Code/WorldGen
git status
```

你会看到一大堆红色的文件,标记为 `Untracked files`,这表示
"Git 看到了新文件,但还没纳入版本管理"。这是正常的。

---

## 第 6 步:**三步提交咒语**(commit + push)

这是核心三步,你以后每次提交都用这三个命令:

### 6.1 `git add`(把改动放进暂存区)

```bash
git add reacthuman_metrics/
```

这告诉 Git:"reacthuman_metrics 里的所有改动我都要提交"。

再用 `git status` 看一下,刚才红色的文件应该变成绿色了,表示
"已暂存,准备提交"。

### 6.2 `git commit`(把改动正式记进本地仓库)

```bash
git commit -m "Add reacthuman_metrics module: evaluation metric suite"
```

引号里的是 **commit message**(提交说明)。
**这是给团队成员看的**,要写得清楚、专业,有助于以后追溯改动。

下面给你三个备选的 commit message,**复制最长那条**,因为
学术合作里 commit message 越详细越好:

**短版:**
```
Add reacthuman_metrics: evaluation metric suite
```

**中版:**
```
Add reacthuman_metrics: dual-track evaluation metric suite

Implements Track 1 (semantic action accuracy, fatal execution rate,
adversarial fool rate) and Track 2 (trajectory error, TTC error,
reachable prediction rate) metrics, along with difficulty-conditioned
aggregation across initial-state, action, and physics axes.
```

**长版(推荐):**
```
Add reacthuman_metrics: dual-track evaluation metric suite

This commit introduces the evaluation metric module for the
ReactHuman benchmark. The module is decoupled from the scene
generator (genesis_scene_generation/) and consumes the on-disk
spec.json format together with JSON-Lines model predictions.

Contents:
- reacthuman.schemas: typed dataclasses for SceneSpec, GroundTruth,
  ModelPrediction, SceneResult, and BenchmarkReport, plus the
  ActionPrimitive / TaskFamily / SafetyLabel enums.
- reacthuman.metrics.track1: semantic_action_accuracy,
  fatal_execution_rate, adversarial_fool_rate.
- reacthuman.metrics.track2: trajectory_error_cm,
  time_to_collision_error, reachable_prediction_rate.
- reacthuman.metrics.scene: per-scene evaluator returning a
  SceneResult.
- reacthuman.metrics.aggregate: corpus-level aggregator returning a
  BenchmarkReport with difficulty-conditioned breakdowns.
- reacthuman.io: spec.json and JSONL prediction loaders.
- reacthuman.cli: 'reacthuman' command-line entry point.

Includes 14 unit tests (all passing) and an end-to-end demo
(examples/run_demo.py) that runs in-memory without external
dependencies.

Mathematical definitions and aggregation protocol are documented
in docs/DESIGN.md; a Chinese quickstart is provided in
docs/QUICKSTART_zh.md.
```

执行 commit 之后会看到类似:
```
[mengyang/physics-benchmark abc1234] Add reacthuman_metrics: ...
 16 files changed, 1284 insertions(+)
```

### 6.3 `git push`(把本地仓库的改动推送到 GitHub)

```bash
git push origin mengyang/physics-benchmark
```

第一次 push 可能会问你确认 GitHub 的 SSH 指纹,输入 `yes` 回车。

看到 `To github.com:BangLab-UdeM-Mila/WorldGen.git` 和一行进度条
说明成功了。

### 6.4 在网页上确认

打开浏览器,访问:

```
https://github.com/BangLab-UdeM-Mila/WorldGen/tree/mengyang/physics-benchmark
```

应该能看到你刚刚添加的 `reacthuman_metrics/` 文件夹。点进去能看到
README,可以预览全部代码。

---

## 第 7 步:在 Slack 上通知团队

提交完之后给 Yizhan、Bang、Jianxin 发个消息:

> Hi @Yizhan @Bang @Jianxin, I just pushed the evaluation metric
> module to `mengyang/physics-benchmark`. It implements both Track 1
> and Track 2 metrics plus the three-axis difficulty conditioning we
> discussed. Path: `reacthuman_metrics/`. README and a Chinese
> quickstart are inside; mathematical definitions are in
> `docs/DESIGN.md`. All 14 unit tests pass.
>
> @Yizhan — two small things I need from your side: (1) confirm the
> field names in `spec.json` match what `reacthuman/io.py` expects,
> and (2) for adversarial scenes, add an
> `extras.appearance_implied_action` field indicating which action
> a model would pick if it only relied on appearance. Both are
> documented in `docs/QUICKSTART_zh.md`. Happy to sync.

---

## 第 8 步:以后改代码再提交的标准流程

每次你修改了代码,无论改了什么,都是这四步:

```bash
# 1. 看看改了什么
git status

# 2. 把改动放进暂存区
git add reacthuman_metrics/   # 或者 git add . 添加所有

# 3. 提交到本地
git commit -m "简短描述这次改了什么"

# 4. 推送到 GitHub
git push origin mengyang/physics-benchmark
```

如果别人(比如 Yizhan)在远程改了东西,你拉下来:

```bash
git pull origin mengyang/physics-benchmark
```

---

## 常见问题排查

### "Permission denied (publickey)"
SSH key 没配好。回到第 2 步重做一遍,特别是 2.3 的测试要通过。

### "Updates were rejected because the remote contains work that you do not have locally"
有人在远程改了东西,你本地落后了。先拉再推:
```bash
git pull origin mengyang/physics-benchmark --rebase
git push origin mengyang/physics-benchmark
```

### "fatal: not a git repository"
你不在 WorldGen 文件夹里。`cd ~/Code/WorldGen` 然后再试。

### 我加错文件了,想撤销 `git add`
```bash
git reset HEAD <文件名>
# 或者撤销所有暂存
git reset HEAD
```

### 我 commit 错了,想改 commit message
```bash
# 如果还没 push:
git commit --amend -m "新的 message"

# 如果已经 push 了,就接受现实,新 commit 一次再 push
```

### 想看看我提交了什么
```bash
git log --oneline -10   # 最近 10 次提交
git show                # 最近一次提交的具体改动
```

---

## 关于保密

Yizhan 在 Slack 里强调了**保密**。具体到操作上:

1. **永远不要把代码 push 到 public fork**。所有 push 都只去
   `BangLab-UdeM-Mila/WorldGen`,不去任何你的个人 fork。
2. **不要在 commit message、issue、PR 里提到具体数字、模型名字、
   API key、内部网址**。
3. **`.gitignore` 已经把 `report_*.json` 排除掉了**,你跑出来的
   实验结果默认不会被提交,这是安全的。如果以后需要提交某些结果,
   要先和团队确认。
4. **不要把仓库 URL 发到任何公开渠道**(微信群聊里也算)。

---

## 一份最简版命令清单(你可以打印出来贴在屏幕边上)

```bash
# 进入仓库
cd ~/Code/WorldGen

# 切到自己的分支(已经在了就跳过)
git checkout mengyang/physics-benchmark

# 拉下别人的最新改动
git pull origin mengyang/physics-benchmark

# 改完代码后,三步走:
git add reacthuman_metrics/
git commit -m "讲清楚改了什么"
git push origin mengyang/physics-benchmark
```

就这些。第一次会觉得很多,做过两三次之后是肌肉记忆。
