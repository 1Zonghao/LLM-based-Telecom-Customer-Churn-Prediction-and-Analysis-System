# 自主数据科学智能体

基于大语言模型（LLM）的微型自主数据科学系统，专为电信客户流失预测场景设计。系统采用 Dual-Track ReAct 架构，结合 LangGraph 状态图与有状态 Jupyter Kernel 沙箱，能够自主完成数据探索、特征工程、模型训练及业务洞察生成，并在关键节点通过 Human-in-the-Loop 断点机制让用户确认方案。

![项目主页](home.png)

---

## 目录

- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [断点确认机制](#断点确认机制)
- [交付物](#交付物)
- [环境变量](#环境变量)
- [许可证](#许可证)

---

## 核心特性

- **Dual-Track ReAct 架构**：语义对话轨（Chat Track）与运行时状态轨（State Track）分离，有效防止长上下文分析中的上下文爆炸。
- **有状态代码执行沙箱**：基于 Jupyter Kernel，变量与模型对象跨轮次持久化，原生支持 stdout、stderr、图表捕获与图像持久化。
- **关键节点断点拦截**：在特征工程前与模型训练前自动挂起，展示方案并支持用户输入修改意见，确认后再执行。
- **Observation 压缩中间件**：DataFrame 输出自动摘要为 shape/dtypes，Traceback 自动截取关键行，防止 LLM 上下文窗口溢出。
- **流式 Gradio 前端**：实时展示对话历史、执行状态、进度面板（阶段/迭代/关键指标）与断点确认面板。
- **自动导出代码历史**：分析完成后自动生成 `.py` 执行脚本与 `.ipynb` Notebook，便于复盘与复现。

---

## 系统架构

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

---

## 技术栈

| 模块 | 技术/库 |
|------|---------|
| 前端 | Gradio |
| 智能体框架 | LangGraph (1.x) |
| LLM API | DeepSeek (兼容 OpenAI SDK) |
| 代码执行沙箱 | Jupyter Kernel (jupyter_client) |
| 数据处理 | Pandas, Scikit-learn, XGBoost, Imbalanced-learn |
| 可视化 | Matplotlib, Seaborn |
| 环境管理 | Docker |

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/1Zonghao/LLM-based-Telecom-Customer-Churn-Prediction-and-Analysis-System.git
cd LLM-based-Telecom-Customer-Churn-Prediction-and-Analysis-System
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
LLM_API_KEY=sk-your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-pro
```

> **注意**：`.env` 已被 `.gitignore` 排除，不会意外提交到仓库。

### 4. 启动服务

```bash
python src/main.py
```

启动后浏览器自动打开 `http://localhost:7860`，界面包含：
- **左侧面板**：数据集上传、任务描述、开始/停止按钮、实时进度观测
- **右侧主区域**：流式对话历史、断点确认面板（含用户反馈输入框）

### 5. 运行测试

```bash
# Phase 1: 沙箱内核测试
python tests/test_sandbox.py

# Phase 2: LangGraph 智能体集成测试
python src/main_agent.py

# Phase 4: 鲁棒性测试
python tests/test_robustness.py
```

---

## 项目结构

```text
.
├── CLAUDE.md                 # 项目技术方案文档
├── README.md                 # 中文说明
├── README-EN.md              # 英文说明
├── requirements.txt          # Python 依赖
├── .env                      # 环境变量（本地配置，勿提交）
├── .gitignore
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # 示例数据集
├── docker/
│   └── Dockerfile            # 无头 Jupyter 沙箱镜像
├── outputs/                  # 自动导出的代码历史（.py / .ipynb）
├── src/
│   ├── main.py               # Gradio 前端入口
│   ├── main_agent.py         # Phase 2 集成测试
│   ├── agent/
│   │   ├── state.py          # AgentState 双轨状态定义
│   │   ├── graph.py          # LangGraph 编译与流式封装
│   │   ├── nodes.py          # reasoning / action / checkpoint 节点
│   │   ├── prompts.py        # SYSTEM_PROMPT / CONTINUE_PROMPT
│   │   └── llm_client.py     # OpenAI 兼容 LLM 客户端
│   ├── sandbox/
│   │   ├── kernel_manager.py # Jupyter Kernel 有状态管理
│   │   └── compressor.py     # 执行结果压缩中间件
│   └── ui/
│       └── app.py            # Gradio UI 与断点交互逻辑
└── tests/
    ├── test_sandbox.py       # Phase 1 沙箱冒烟测试
    └── test_robustness.py    # Phase 4 边界与自修复测试
```

---

## 断点确认机制

系统在两个关键节点通过 `interrupt_before` 自动挂起，等待用户确认：

1. **特征工程前（`checkpoint_fe`）**
   - 智能体输出 EDA 总结与特征处理方案（清洗策略、编码方式、重采样方法）
   - 用户可在反馈框中补充要求（如"改用 SMOTE 而非 class_weight"、"对 TotalCharges 做对数变换"）
   - 点击「确认执行」后，反馈内容会注入后续对话，智能体据此调整并执行

2. **模型训练前（`checkpoint_model`）**
   - 智能体确认算法列表与重采样策略
   - 用户可修改模型选型或评估方式

点击「跳过后续确认 / 自动完成」后，系统将设置 `auto_confirm=True`，后续所有断点自动通过，不再弹出确认面板。

---

## 交付物

分析完成后，系统自动在 `outputs/` 目录生成：

- **`analysis_YYYYMMDD_HHMMSS.py`**：完整的代码执行历史（带步骤注释的 Python 脚本）
- **`analysis_YYYYMMDD_HHMMSS.ipynb`**：标准 Jupyter Notebook（含 Markdown 任务描述与可执行代码单元）

---

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `LLM_API_KEY` | 是 | — | DeepSeek（或其他 OpenAI 兼容服务商）API 密钥 |
| `LLM_BASE_URL` | 否 | `https://api.deepseek.com` | API 基础地址 |
| `LLM_MODEL` | 否 | `deepseek-v4-pro` | 模型名称 |

---

## 许可证

MIT License © 2025 1Zonghao
