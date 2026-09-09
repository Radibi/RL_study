# Hand2 microwave manipulation RL

一个基于 [MuJoCo](https://mujoco.readthedocs.io/) 和 Gymnasium 的灵巧操作训练起点：7-DoF 机械臂驱动舞肌 Hand2 风格的五指末端执行器，利用多个指尖触觉传感器完成“打开微波炉门并将鸡腿放入腔体”的长时程任务。

> 本仓库提供的是可训练的简化研究模型，而非舞肌 Hand2 的厂家标定模型。替换为实机 CAD/URDF 参数前，请完成碰撞、关节限位、传动和安全验证。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev]'
```

## 训练与观察

```bash
python scripts/train.py --timesteps 1_000_000 --n-envs 4
python scripts/play.py --model runs/ppo_hand2_microwave/final_model.zip
```

训练使用 PPO；动作是 7 个机械臂关节和 10 个手指关节的位置增量，观测包含关节状态、鸡腿/门状态、末端相对位姿、5 路指尖触觉值以及任务阶段。奖励采用稠密的分阶段整形：接近把手、开门、接近鸡腿、抓取、移动到腔体和释放。

## 项目结构

* `hand2_microwave/assets/scene.xml`：MuJoCo 场景、7-DoF 机械臂、Hand2 风格五指、微波炉、鸡腿和触觉站点。
* `hand2_microwave/env.py`：Gymnasium 环境与阶段奖励。
* `scripts/train.py`：向量化 PPO 训练入口。
* `scripts/play.py`：加载策略并可视化回放。

## 设计说明

`touch` 传感器直接读取每个指尖 site 的接触法向力；环境将其裁剪并归一化后放入观测。门铰链、鸡腿 free joint 和末端接触均由 MuJoCo 求解。成功条件是门打开、鸡腿位于腔体目标区域且速度低于阈值。

若使用厂商 Hand2 模型，请保留 site 名称 `tactile_*`、执行器顺序和 17 维动作接口，或同步修改 `env.py` 中的名称映射。
