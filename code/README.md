# 方案1主代码

这里只包含方案1的主模型、数据处理、训练、推理与必要工具，不含探索、对照、消融、实验日志或结果缓存。

| 文件 | 用途 |
| --- | --- |
| model.py | 384维、18比特角色—结构纠缠主模型，49,361个训练参数 |
| quantum_ops.py | 门操作、四来源匹配与相位通道 |
| data.py | 原始四路384维语义、结构与角色输入 |
| train.py | 主模型的排序损失、优化器、开发选点与训练入口 |
| inference.py | 主模型权重加载与候选评分 |
| metrics.py | 排序指标及训练权重 |
| config.py | 数据、权重、原始训练配置和新输出路径 |
| utils.py | 文件哈希与输出写入 |

历史文件夹依赖已合并到这些模块，运行时不导入历史实验的Python代码。数据与已训练权重继续读取原WWW/验证目录中的原始文件，不复制进本代码目录。路径由config.py或QAFFORD_WWW、QAFFORD_DATA、QAFFORD_ROLES、QAFFORD_ARTIFACTS环境变量指定。

加载模型：`from inference import load_grounder, score_encoded`。输入streams形状为[batch, candidate, 4, 384]，layout为[batch, candidate, 10]，role为[batch, candidate]；四个来源依次为T/D/C/B，归一化与训练保持一致。

训练入口：`python train.py --seed 0 --stage development`。仅当显式运行命令时训练；输出默认写到上层运行输出/方案1，不覆盖原始封存实验。final阶段需传入开发集固定的--selected和--horizon。本次整理没有启动训练或重新选择测试配置。
