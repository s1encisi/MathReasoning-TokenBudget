# MathReasoning-TokenBudget

**面向数学推理的 SFT / GRPO 训练与回答长度控制实验基座。** 项目把训练入口、可组合奖励函数、数据生成和评测配置放在同一条工作流中，便于检查正确率与回答长度之间的关系。

训练与评测实现来自 [Hugging Face Open-R1](https://github.com/huggingface/open-r1)，本仓库在其基础上维护中文命令行、日志、错误信息和格式化支持。目前交付的是可配置的实验代码与本地化实现；尚未发布本仓库独立训练的模型或 token 节省实验结果。

## 从哪里看代码

| 能力 | 实现入口 | 关键设计 |
| --- | --- | --- |
| 监督微调与 GRPO | [sft.py](src/open_r1/sft.py)、[grpo.py](src/open_r1/grpo.py) | 使用 TRL Trainer，模型、数据和训练参数由 YAML 配置组织 |
| 正确性与长度奖励 | [rewards.py](src/open_r1/rewards.py) | 数学答案验证、输出格式、推理步骤、重复惩罚及长度相关奖励可组合 |
| 数据处理 | [data.py](src/open_r1/utils/data.py)、[scripts](scripts) | 数据集混合、固定种子采样、去污染及基于通过率的筛选 |
| 生成与评测 | [generate.py](src/open_r1/generate.py)、[Makefile](Makefile) | Distilabel 生成链路，LightEval / vLLM 评测入口 |
| 中文使用体验 | [i18n](src/open_r1/i18n)、[测试](tests/test_i18n.py) | JSON 语言资源、中英切换、中文终端宽度与数字/日期格式化；核心本地化模块仅依赖标准库 |

训练数据经配置加载后进入 SFT 或 GRPO；GRPO 根据 `reward_funcs` 和 `reward_weights` 组合奖励，检查点与评测产物写入本地输出目录。分布式配置见 [recipes/accelerate_configs](recipes/accelerate_configs)，集群脚本见 [slurm](slurm)。

### 长度与 token 的计量区别

现有 `length` 和 `cosine` 奖励使用回答字符串长度；`soft_overlong_punishment` 使用 `completion_ids` 的 token 数量。比较 token 成本时应使用 token 口径，不能把字符数变化直接解释为 token 节省。默认数学 GRPO 示例组合的是 `accuracy`、`format` 和 `tag_count`，并未默认开启长度惩罚。

## 快速检查：无需 GPU

使用 Python 3.10.9 或以上版本，在仓库根目录运行：

```bash
# Linux / macOS
PYTHONPATH=src python -m unittest tests.test_i18n -v
```

```powershell
# Windows PowerShell
$env:PYTHONPATH = "src"
python -m unittest tests.test_i18n -v
```

语言默认为简体中文；设置 `OPENR1_LOCALE=en_US` 可切换为英文，`OPENR1_LOCALE=zh_CN` 切回中文。语言资源随 Python 包分发。

## 训练环境与配置

GPU 训练沿用 Open-R1 的 Linux / CUDA 工作流。依赖约束由 [setup.py](setup.py) 和 [Makefile](Makefile) 维护，包括 PyTorch 2.6.0、Transformers 4.52.3、TRL 0.18.0、Accelerate 1.4.0，以及 vLLM / FlashAttention。请按实际 CUDA、显存和 GPU 数量准备环境，再选择配置：

- [1.5B 数学 GRPO 示例](recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml)：DeepSeek-R1-Distill-Qwen-1.5B 与 OpenR1-Math-220k，最大回答长度 2048。
- [7B 蒸馏 SFT 配置](recipes/OpenR1-Distill-7B/sft/config_distill.yaml)：Mixture-of-Thoughts 数据；原配置针对 8 × H100 80GB，不能按普通单卡配置理解。
- [训练配置说明](recipes/README.md)与 [Open-R1 使用文档](https://github.com/huggingface/open-r1#installation)：依赖安装、训练启动及评测参数。

示例配置启用了 `push_to_hub` 和 W&B 记录；仅在本地实验时，应先将 `push_to_hub` 设为 `false`，并调整 `report_to`、输出路径、批量大小与生成长度。模型权重和训练数据需另行获取。

## 已验证范围

本地检查通过 23 个 i18n 单元测试，以及仓库约定的 Ruff、isort 和 Flake8 检查；Python 文件语法解析通过。测试覆盖语言资源完整性、语言切换、占位符、中文排版与格式化。完整训练及模型效果评测需要额外 GPU 环境，本仓库尚未给出相应实测指标。

## 来源与许可

派生基线为 Open-R1 提交 `1416fa0cf21595d2083b399a2a0bbddd7f6e9563`，保留上游作者版权声明与 Git 历史。代码采用 [Apache-2.0](LICENSE)；模型和数据使用各自的许可证。上游模型成绩属于上游项目。
