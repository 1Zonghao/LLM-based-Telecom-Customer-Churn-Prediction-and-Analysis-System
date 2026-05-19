# 微型自主数据分析智能体 - 技术开发方案 (Optimized Edition)

## 1. 项目概述
本项目旨在构建一个基于大语言模型（LLM）的微型自主数据科学智能体，对应清华大学数据库组“自主数据科学系统”的构想。智能体通过调度有状态的代码执行沙箱，能够自主完成数据探索、特征工程、模型训练及业务洞察生成。

**核心工程哲学**：
开发过程需遵循“先谋后动”与“抓住主要矛盾”的原则。在本项目中，核心业务矛盾是**电信流失场景中的非对称损失（漏报代价远大于误报）**，核心系统矛盾是**长时间数据分析产生的数据噪音与 LLM 有限上下文窗口之间的冲突**。整个架构将围绕解决这两个矛盾展开。

## 2. 系统架构 (Dual-Track ReAct 架构)

```text
┌─────────────────────────────────────────────────────────────┐
│                       前端 (Gradio)                           │
│  - 数据集上传  - 自然语言指令  - 关键节点断点确认 (Human-in-the-Loop) │
└───────────────────────┬───────────────▲─────────────────────┘
                        │ 指令/干预      │ 流式思考/图表展示
┌───────────────────────▼───────────────┴─────────────────────┐
│                    智能体核心 (LangGraph)                     │
│ ┌──────────────────────────┐   ┌──────────────────────────┐ │
│ │  Semantic Chat Track     │   │  Runtime State Track     │ │
│ │ (过滤后的Observation、推理) │   │ (数据Schema、当前最佳指标)  │ │
│ └──────────────────────────┘   └──────────────────────────┘ │
│          - 状态路由 (State Graph) & 断点管理 (Breakpoints)      │
└───────────────────────┬───────────────▲─────────────────────┘
                        │ 生成代码       │ 执行结果/异常/图片引用
┌───────────────────────▼───────────────┴─────────────────────┐
│                 有状态代码执行沙箱 (Jupyter Kernel)             │
│   - 持久化内存空间 (变量、模型对象跨轮复用)                        │
│   - 自动捕获 stdout、stderr、display_data (图表 base64)         │
└─────────────────────────────────────────────────────────────┘

```

## 3. 技术栈选型

| 模块 | 技术/库 | 说明 |
| --- | --- | --- |
| 前端 | Gradio | 支持流式输出、图表渲染及断点交互设计 |
| 智能体框架 | LangGraph | 利用其 State Graph 实现严谨的控制流与断点挂起 |
| LLM API | deepseek-v4-pro | 核心推理与代码生成引擎 |
| 代码执行沙箱 | **Jupyter Kernel (gRPC 或 E2B Sandbox)** | **核心优化**：维持有状态内存，避免重复加载数据，原生支持富文本和图像捕获 |
| 数据处理与建模 | Pandas, Scikit-learn, XGBoost, Imblearn | 容器内预置标准数据科学全家桶 |
| 环境管理 | Docker | 物理隔离代码执行环境，防止越权访问 |

## 4. 业务场景与主要矛盾分析

* **数据集**：Kaggle Telco Customer Churn (7043 行 × 21 列)
* **业务导向**：目标变量 `Churn`（流失比例约 26.5%）。由于挽留老客户的成本低于拉新成本，系统的**主要评估指标必须锚定 Recall（召回率）和 F1-score**。
* **预处理关键**：`TotalCharges` 空白符处理；`SeniorCitizen` 类型转换；必须引入 SMOTE 或类权重调整。

## 5. 核心功能模块设计

### 5.1 双轨记忆与状态管理 (Dual-track Memory)

防范“上下文崩溃”（Context length explosion）的核心机制：

```python
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    task: str                         # 核心目标
    data_profile: dict                # State轨：动态元数据字典 (如当前列名、数据分布、最佳评估分数)
    messages: Annotated[List[dict], operator.add] # Chat轨：压缩后的历史记录 (仅留核心报错或摘要)
    code_history: List[str]           # 完整代码轨迹 (用于复盘或生成完整 Notebook)

```

* **Observation 压缩**：沙箱返回的大型 DataFrame 或千行报错，必须经过中间件压缩（如只截取 Traceback 最后三行，或 DataFrame 的 `shape` 和 `dtypes`）后再写入 `messages`。

### 5.2 有状态工具调用 (Stateful Sandbox)

摒弃 `subprocess` 或单次 `exec()`，采用 Jupyter Kernel 会话机制：

1. LLM 发出 Python 代码。
2. LangGraph 节点通过 Jupyter 协议发送 `execute_request`。
3. 自动解析 `iopub` 消息。若包含图表绘制（`image/png`），沙箱自动持久化为本地图片，并返回 Markdown 图片标签给 LLM，实现多模态展示。

### 5.3 关键节点断点拦截 (Human-in-the-Loop)

在 LangGraph 中配置 `interrupt_before`，实现人机协同的灰度控制：

* **`before_feature_engineering` 节点**：智能体输出 EDA 总结和特征处理方案（清洗策略、编码方式）后挂起，用户在前端确认无误或补充提示后再执行。
* **`before_model_training` 节点**：智能体确认算法列表和重采样策略后挂起，供业务人员二次确认。

## 6. 渐进式开发路线 (Iterative Roadmap)

**Phase 1: 基础设施建设 —— 有状态沙箱基石**

* 构建 Docker 镜像，配置无头 Jupyter 环境。
* 封装沙箱通信接口，确保能连续执行两段代码共享同一变量（如第一段定义 `df`，第二段打印 `df.head()`）。
* 实现富文本（HTML/Images）的截获与本地转存。

**Phase 2: 智能体神经中枢 —— Graph 与双轨记忆**

* 搭建 LangGraph 骨架，定义 `AgentState`。
* 实现 `reasoning_node` 和 `action_node`。
* 编写核心中间件：实现执行结果的**压缩机制**（提取核心摘要而非长文本）。

**Phase 3: 领域知识注入与人机交互**

* 编写系统提示词（System Prompt），明确电信流失场景的业务关注点（召回率优化）。
* 部署 Gradio 前端，集成 LangGraph 运行流。
* 接入断点机制（Breakpoints），在前端实现确认按钮。

**Phase 4: 边界测试与鲁棒性打磨**

* 注入常见异常（拼写错误列名、内存溢出），测试智能体的自我 Debug 循环。
* 微调提示词，确保生成的最终业务报告具有高可解释性。

## 7. 评估验收指标

1. **工程健壮性**：
* 连续进行多步长上下文数据处理时，上下文 Token 不超过阈值（无 Lost in the Middle 现象）。
* 沙箱内变量能正确持久化和覆盖。


2. **分析质量**：
* 目标数据测试集 F1-score (Churn) ≥ 0.65，AUC ≥ 0.80。
* 特征工程覆盖率：正确执行缺失值插补、Categorical 编码、类别不平衡处理。


3. **交付规范**：
* Docker 容器一键启动。
* 分析结束后可提供完整的、格式规整的代码执行历史记录。



```

```