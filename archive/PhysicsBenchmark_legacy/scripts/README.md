# WorldGen Simulation Pipeline
## BangLab · UdeM · Mila

---

## 文件夹结构（你会有这些文件）

```
worldgen/
├── scripts/
│   ├── 00_check_setup.sh        ← 第0步：检查工具
│   ├── 01_generate_objects.py   ← 第1步：Blender生成物体
│   ├── 02_generate_urdfs.py     ← 第2步：生成URDF文件
│   └── 03_run_simulation.py     ← 第3步：跑仿真
├── objects/                     ← 自动生成：.obj mesh文件
├── urdfs/                       ← 自动生成：.urdf文件
└── logs/                        ← 自动生成：图表+数据JSON
```

---

## 第0步：安装所有工具（只需做一次）

打开终端，复制粘贴这一整块命令：

```bash
# 安装 Python 包
pip3 install pybullet trimesh numpy scipy matplotlib tqdm

# 安装 Blender（Ubuntu/Debian Linux）
sudo apt update && sudo apt install -y blender
```

如果你是 **macOS**：
- Blender: 去 https://www.blender.org/download/ 下载，拖入 Applications
- 其余 pip 命令一样

验证是否全部安装好：
```bash
bash scripts/00_check_setup.sh
```
看到全部 [OK] 才继续。

---

## 第1步：Blender 生成三个研究级物体

在 `worldgen/` 目录下运行：

```bash
blender --background --python scripts/01_generate_objects.py -- --output ./objects
```

**会生成什么：**
| 文件 | 是什么 |
|------|--------|
| `objects/sphere.obj` | 橡胶球（测滚动接触） |
| `objects/sphere_col.obj` | 球的碰撞简化mesh |
| `objects/bevel_box.obj` | 木质倒角方块（测摩擦滑动） |
| `objects/bevel_box_col.obj` | 方块碰撞mesh |
| `objects/l_bracket.obj` | 铝制L形件（测非对称惯性张量） |
| `objects/l_bracket_col.obj` | L形件碰撞mesh |

---

## 第2步：自动生成 URDF 文件

```bash
python3 scripts/02_generate_urdfs.py --objects_dir ./objects --urdfs_dir ./urdfs
```

**会打印什么（正常输出）：**
```
[Processing] sphere
  mass   = 0.0524 kg
  CoM    = [0.0000, 0.0000, 0.0000]
  Ixx    = 2.09e-05  Iyy=2.09e-05  Izz=2.09e-05
  → URDF: ./urdfs/sphere.urdf
```

---

## 第3步：跑 PyBullet 仿真

```bash
python3 scripts/03_run_simulation.py --urdfs_dir ./urdfs --logs_dir ./logs
```

**会生成：**
- `logs/e1_freefall.png` — 球自由落体的轨迹+能量图
- `logs/e2_incline.png`  — 方块斜面滑动速度图（验证摩擦系数）
- `logs/e3_rotation.png` — L形件扭矩自由旋转（Dzhanibekov效应）
- `logs/benchmark_results.json` — 所有数据

---

## 三个实验的物理意义

### E1：自由落体+弹跳（球）
验证恢复系数 e。理论上第n次弹跳高度：
```
h_n = e^(2n) × h_0
```
用来检验 LLM 生成的代码中 restitution 参数是否正确。

### E2：斜面滑动（方块）
在20°斜面上，理论加速度：
```
a = g × (sin20° − μ × cos20°)
```
从仿真数据拟合 a，反推摩擦系数 μ，和设定值对比。

### E3：扭矩自由旋转（L形件）
L形件的三个主惯量轴不对称，
绕中间轴旋转是不稳定的（Dzhanibekov效应）。
验证 URDF 中 ixx/ixy/ixz 等惯性张量参数是否正确。

---

## 常见报错

| 报错信息 | 解决方法 |
|----------|----------|
| `blender not found` | 用 `apt install blender` 或去官网下载 |
| `No module named 'pybullet'` | `pip3 install pybullet` |
| `mesh is not watertight` | 正常警告，不影响运行，会自动用 convex hull |
| `No module named 'trimesh'` | `pip3 install trimesh` |
| `.obj not found` 跳过某实验 | 先确认第1步Blender成功运行 |

---

*Generated for BangLab WorldGen Pipeline · March 2026*
