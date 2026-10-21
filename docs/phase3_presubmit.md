# Phase 3：Pre-submit 强制自检节点

## 1. 目标

Phase 2 的结论是：**verifier 反馈里的自检代码会被执行，任务提示里的不会。**
但它仍然依赖智能体“愿意执行”自检，且每次都要消耗一次 verifier 重试。

Phase 3 把自检变成图结构中的**强制节点**：

```
reasoning --finish--> pre_submit --pass--> verifier --pass--> END
                         |                    |
                         +--fail--> reasoning <+
```

- `pre_submit` 不调用 LLM；
- 它直接从 artifact contract 生成自检代码，在同一个 Jupyter kernel 里执行；
- 自检失败：把错误和自检代码作为反馈注入，回到 reasoning 修复；
- 自检通过：才允许进入 verifier；
- verifier 仍然保留，作为最终交付闸门。

## 2. 实现

- `src/agent/verifier.py`
  - `build_submission_self_check_code(contract)`：生成可直接执行的原始代码；
  - `build_submission_self_check(contract)`：生成带 Markdown 围栏的版本（提示词/反馈用）；
  - `format_pre_submit_feedback(...)`：把沙箱错误转成修复指令。
- `src/agent/nodes.py`
  - `make_pre_submit_node(km)`：执行自检代码；
  - `pre_submit_router`：`repair -> reasoning` / `verify -> verifier`；
  - 状态新增 `pre_submit_result`、`pre_submit_history`、`pre_submit_failures`。
- `src/agent/graph.py`
  - reasoning 的 `finish` 改为进入 `pre_submit`；
  - `pre_submit` 通过后才进入 `verifier`。
- `src/agent/episode.py`
  - 结果中输出 pre-submit 相关字段；
  - 硬上限从 `max_iter + 4` 调整为 `max_iter + 8`。

## 3. 实验结果（Adult，2 轮，skill-mode off）

| 轮次 | verified | verifier 重试 | pre-submit 失败 | 自检结果 | AUC | F1 | Tokens | 耗时 |
|---|---|---|---|---|---|---|---|---|
| 1 | ✅ | **0** | 1 | fail → pass | 0.9303 | 0.7423 | 73,944 | 356 s |
| 2 | ✅ | **0** | 1 | fail → pass | 0.9295 | 0.7369 | 163,360 | 488 s |

关键观察：

- 两轮的 `verification_history` 都只有一次 `passed=True`；
- 第一次最终报告被 `pre_submit` 拦截，错误是“预测文件不存在”；
- 智能体收到反馈后，在修复代码中运行了自检（`code_history` 中有 1 个 `SELF-CHECK` 代码块）；
- 第二次提交通过 `pre_submit`，随后 verifier 一次通过；
- **verifier 重试从 Phase 2 的 1 次降到 0 次。**

## 4. 与 Phase 2 的对比

| 版本 | 自检触发方式 | verifier 重试 | 是否依赖智能体主动执行 |
|---|---|---|---|
| Phase 2 v2 | verifier 失败反馈 | 1 | 是 |
| Phase 3 pre-submit | 图节点强制 | **0** | **否** |

## 5. 局限

- 单次运行，Token/耗时方差仍大（第 2 轮 163k tokens 明显高于第 1 轮）；
- pre-submit 只检查“交付物格式/存在性”，不检查建模质量、泄漏、验证方案；
- 自检代码由契约生成，尚未把 skill library 的检查函数动态组合进来。

## 6. 下一步

1. **Skill library v2**：把技能存成检查函数/代码模板，由 pre-submit 节点按需组合执行；
2. 把执行错误（沙箱报错）也纳入技能沉淀；
3. 多轮重复 + held-out 数据集，验证稳定性与迁移；
4. pre-submit 增加建模质量检查（泄漏、验证方案、分组切分）。
