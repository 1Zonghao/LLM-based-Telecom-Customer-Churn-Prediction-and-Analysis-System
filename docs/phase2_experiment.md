# Phase 2 实验记录：Verifier 驱动的可验证自改进

## 1. 目标

Phase 1 让智能体学会“检查自己的作业”（交付物契约 + verifier + 修复回路）。
Phase 2 要回答的问题是：

> **智能体能否把失败经验带到下一次任务，并在新任务上表现更好？**

为此增加了两层：

- **Skill library**：从 verifier 失败中沉淀可复用经验；
- **Experiment runner**：多数据集、多轮次、skill on/off 对比，记录通过率、重试次数、Token、耗时和指标。

## 2. 架构

```
episode runner
  ├── prepare_dataset()      固定训练/留出切分
  ├── SkillLibrary.retrieve() 检索过往经验（按轮次冻结）
  ├── run_episode()           verifier-gated 智能体
  ├── score_predictions()     AUC / F1 / Recall
  └── SkillLibrary.add_from_verification()  沉淀新经验
```

关键设计：

- **每轮开始时冻结技能库**：本轮新学到的经验只从下一轮生效，避免同轮污染轮次对比；
- **技能有两种形态**：
  1. 文字教训（Phase 2 v0/v1）；
  2. 可执行自检代码（Phase 2 v2）；
- **verifier 是最终闸门**：只有交付物通过检查，episode 才算完成。

## 3. 实验设置

- 数据集：Adult（26,048 训练 / 6,513 留出）、Bank（32,950 / 8,238）；
- 固定分层切分，目标为二分类，指标 ROC-AUC（首要）+ F1 + Recall；
- 轮次：2 轮；每轮训练/留出切分固定；
- 对照：`skill-mode on`（检索并沉淀经验）vs `off`（不检索、不沉淀）；
- 模型/提示词/工具链相同，唯一变量是技能注入。

## 4. 结果

### 4.1 Adult：三代技能注入方式对比

| 版本 | 注入位置 | 智能体是否执行自检 | 失败验证次数 | 结果 |
|---|---|---|---|---|
| v0 | 任务提示中的文字教训 | 否 | 2 | 仍重复同样的错误 |
| v1 | 任务提示中的可执行自检代码 | 否（0 个代码块） | 2 | 第二轮直接失败 |
| v2 | **verifier 失败反馈中的自检代码** | **是（每个 episode 1 个代码块）** | **1** | 一次修复后通过 |

v2 的典型过程：

1. 第一次验证：交付物缺失；
2. verifier 反馈中附带可执行自检代码；
3. 智能体运行自检代码，一次性生成预测/报告并检查列名、行数、概率范围；
4. 第二次验证通过。

### 4.2 v2 on vs off（Adult，2 轮）

| 指标 | on | off |
|---|---|---|
| 通过率 | 100% | 100% |
| 平均失败验证次数 | 1.0 | 1.0 |
| 平均 Token | 87,854 | 87,008 |
| 平均耗时 | 315 s | 233 s |
| 平均 AUC | 0.9318 | 0.9308 |
| 平均 F1 | 0.7434 | 0.7416 |

**结论：在 verifier 反馈自检已经存在的情况下，额外的 skill library 注入没有带来可测量的收益。** 主要收益来自 verifier 反馈，而不是技能库本身。

### 4.3 早期 pilot（v0，Adult + Bank）

- 文字教训没有稳定降低重试或 Token；
- Bank 的 `duration` 是泄漏特征，导致 AUC 波动很大（0.81–0.94），该组结果不作为有效对比；
- 已在数据准备中增加 `drop_columns=["duration"]`，后续实验需用 `--force-data` 重建 Bank 切分。

## 5. 发现

1. **反馈通道比知识通道更有效。** 智能体会执行 verifier 反馈里的代码，但会忽略任务提示里的同样代码。
2. **被动记忆不够。** 把经验写成文字或代码放进 prompt，不能保证执行；必须进入 verifier/图结构的强制路径。
3. **verifier 是当前系统最有效的自改进杠杆。** 它把“智能体以为完成”变成“系统确认完成”，并给出可执行的修复指令。
4. **单次运行方差很大。** 同一配置的 Token/耗时在不同轮次可差 2 倍；需要多次重复和更干净的对照。

## 6. 局限

- 每个配置只跑了 1 次，结论是方向性的，不是统计显著的；
- 只测了 Adult（以及受泄漏影响的 Bank），没有 held-out 数据集迁移实验；
- 技能库目前只从 verifier 失败中学习，没有从执行错误/执行成功中学习；
- 自检代码仍依赖智能体“愿意执行”，尚未做成图结构中的强制节点。

## 7. 下一步

1. **pre-submit 强制节点**：在图里拦截 `finish`，自动执行自检代码；失败则回到 reasoning。这样自检不再依赖提示词。
2. **技能库 v2**：把技能存成“检查函数/代码模板”，由 pre-submit 节点自动执行，而不是注入提示词。
3. **从执行错误学习**：把沙箱报错也纳入技能沉淀（例如特征列不匹配、内存不足）。
4. **更严谨的实验**：3–5 次重复、中位数/置信区间、干净的 Bank、Diabetes held-out 迁移。
5. **多数据集迁移**：在 Adult/Bank 上学到的交付检查，迁移到 Diabetes/Telco 是否仍然有效。

## 8. 复现命令

```bash
# 准备数据（Bank 已移除 duration 泄漏特征时使用 --force-data）
python src/experiment_runner.py --datasets adult --rounds 2 --skill-mode on --reset-skills --provider anthropic

# 对照
python src/experiment_runner.py --datasets adult --rounds 2 --skill-mode off --provider anthropic

# 对比
python src/compare_experiments.py --on-dir outputs/experiments/<on> --off-dir outputs/experiments/<off>
```
