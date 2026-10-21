# Phase 4：Skill v2 —— 把技能变成可执行的 pre-submit 检查

## 1. 目标

Phase 2/3 的结论：

- 文字经验放进任务提示会被忽略；
- verifier 反馈里的自检代码会被执行；
- pre-submit 节点能把自检变成图结构里的强制闸门。

Phase 4 要验证的是：

> **从失败中沉淀的技能，能否作为可执行检查，在后续任务中提前拦住同类问题？**

## 2. 实现

### 2.1 技能激活

- `AgentState` 新增 `active_skill_keys`；
- 实验 runner 检索技能后，把技能 key 传给 `run_episode`；
- pre-submit 节点只执行被激活技能的检查。

### 2.2 可执行检查

`src/agent/verifier.py` 新增 `run_skill_checks(state)`：

- `group_cv`：检查代码里是否使用 GroupKFold / StratifiedGroupKFold / GroupShuffleSplit / groups=；
- `forbidden_features`：读取特征清单，检查是否使用了禁用/泄漏特征；
- `probability_not_hard_labels`：检查预测列是否只有 0/1 两个唯一值。

pre-submit 节点把基础自检（交付物格式）和技能检查合并；任一失败都会拦截最终报告。

### 2.3 Feature Manifest

静态扫描代码不可靠：智能体可能把泄漏特征放进特征列表、或用 `pd.get_dummies` 隐式使用，代码里不一定出现列名。

因此契约新增 `feature_manifest.json`：

```json
{
  "features": ["age", "job", "campaign", "..."]
}
```

- 任务要求智能体写出“最终模型实际使用的全部特征名”；
- verifier 和 pre-submit 都按这份清单检查 `forbidden_features`；
- 清单缺失或为空会直接失败。

## 3. 实验与结果

### 3.1 Group CV（adult_grouped / adult_grouped_site）

| 数据集 | 分组列 | 智能体行为 | verifier 重试 |
|---|---|---|---|
| adult_grouped | `group_id` | 主动使用 StratifiedGroupKFold | 0 |
| adult_grouped_site | `site_code`（不提示） | 仍主动使用 StratifiedGroupKFold | 0 |

结论：**智能体本身就会做分组验证**。`group_cv` 技能没有被触发，也没有必要触发。
这说明 skill library 的边际价值不在“智能体已经会做”的检查上。

### 3.2 Bank 泄漏特征（bank_blind）

数据：Bank Marketing，`duration` 改名为 `call_length`，契约禁止使用。
任务提示不告诉智能体这是泄漏特征。

| 版本 | 检查方式 | 结果 |
|---|---|---|
| 无 manifest | 扫描最后一段代码 | 漏检：智能体使用了 `call_length`，AUC 0.9558，verifier 放行 |
| 有 manifest | 读取特征清单 | 检出：verifier 报 `使用了禁用/泄漏特征: ['call_length']`；修复后 AUC 0.8178 |

结论：**manifest 是必要的**。静态代码扫描会被隐式使用绕过；特征清单能可靠地暴露最终特征集。

### 3.3 技能复用（on vs off）

第一轮 verifier 失败后，`forbidden_features` 被写入 skill library。
后续 episode 激活该技能，pre-submit 在最终报告前读取 manifest 并检查禁用特征。

| 运行 | 注入技能 | pre-submit 技能拦截 | verifier 重试 | Tokens | 耗时 | 最终 AUC |
|---|---|---|---|---|---|---|
| on（独立预算） | `forbidden_features` | **是**（检出自 `call_length`） | **0** | **162,524** | 524.4 s | 0.8196 |
| off | 无 | 否 | 1 | 245,175 | 415.9 s | 0.8094 |

关键观察：

- on 组的 `pre_submit_history` 第一条同时报告了“预测文件不存在”和
  `skill:forbidden_features` 失败“使用了禁用/泄漏特征: `call_length`”；
- 智能体随后移除泄漏特征，第二次 pre-submit 通过，verifier **一次通过（0 重试）**；
- off 组没有技能检查，泄漏只能在 verifier 阶段被发现，因此 verifier 重试 1 次；
- 最终 AUC 从 0.9558 降到约 0.82，说明泄漏确实被消除；
- on 组比 off 组少用约 **34% Token**；耗时在单次运行中反而更高（pre-submit 多了一轮修复），
  需要多轮重复才能判断时间收益。

### 3.4 预算耦合（已修复）

on 组的 pre-submit 一共失败 2 次：

1. 预测文件缺失（基础自检）；
2. manifest 检出 `call_length`（技能检查）。

当时 pre-submit 和 verifier 共用 `max_verification_retries=2`，两次失败后预算耗尽，
pre-submit 把控制权交给 verifier；verifier 又拦了一次，所以 verifier 重试数为 1。

修复：新增独立的 `max_pre_submit_failures`（默认 3），pre-submit 不再和 verifier 抢预算。
独立预算后的 on 运行：`pre_submit_failures=1`、`verifier_retries=0`，技能检查失败后
继续回到 reasoning 修复，verifier 一次通过。

## 4. 发现

1. **智能体在通用过程检查上已经很强**：分组验证、交付格式、概率输出等，base verifier + pre-submit 已经覆盖。
2. **skill library 的价值在“契约特有、智能体不会主动做”的检查上**：泄漏特征、业务禁用字段、数据质量约束。
3. **静态代码扫描不足以做负向检查**：必须依赖结构化的特征清单（或内核探针）。
4. **基础设施稳定性是实验的一部分**：LLM API 的 SSL 瞬断会让 episode 直接失败，需要重试/降级策略。

## 5. 下一步

1. 用内核探针读取模型的真实 `feature_names_in_`，替代“信任 manifest”；
2. 增加 off 对照（无技能）和多轮重复，量化技能复用的收益；
3. 把执行错误（沙箱报错、特征列不匹配）也纳入技能沉淀；
4. 给 LLM 调用增加网络重试/退避，避免瞬断毁掉整个 episode。
