import operator
from typing import Annotated, List, TypedDict


class AgentState(TypedDict):
    # 用户核心任务描述
    task: str
    # 数据集路径
    data_path: str
    # State Track: 动态元数据（列名、类型、分布、最佳评估分数）
    data_profile: dict
    # Chat Track: 压缩后的对话历史
    messages: Annotated[List[dict], operator.add]
    # 完整代码轨迹（用于复盘或生成 Notebook）
    code_history: Annotated[List[str], operator.add]
    # 当前迭代轮次
    iteration: int
    # 路由指令: "execute" | "checkpoint_fe" | "checkpoint_model" | "finish"
    next_action: str
    # 最终分析报告
    final_report: str
    # 当前分析阶段: "eda" | "feature_engineering" | "model_training" | "reporting"
    phase: str
    # 待用户确认的方案文本
    pending_proposal: str
    # 用户在断点处输入的补充提示/修改意见
    user_feedback: str
    # 连续相同错误计数（用于检测死循环）
    consecutive_errors: int
    # Token 消耗累计
    token_usage: dict
    # 分析开始的 ISO 时间戳
    started_at: str
    # 领域插件: "generic" | "telco_churn"
    domain: str
    # 交付物契约（为空时验证器自动通过，兼容旧调用方）
    artifact_contract: dict
    # 最近一次交付验证结果
    verification_result: dict
    # 已执行的验证重试次数
    verification_retries: int
    # 允许的最大验证重试次数
    max_verification_retries: int
    # 是否通过交付验证
    verification_passed: bool


def init_state(
    task: str,
    data_path: str,
    domain: str = "generic",
    artifact_contract: dict | None = None,
    max_verification_retries: int | None = None,
) -> AgentState:
    contract = artifact_contract or {}
    retries = (
        max_verification_retries
        if max_verification_retries is not None
        else int(contract.get("max_verification_retries", 2))
    )
    return {
        "task": task,
        "data_path": data_path,
        "data_profile": {},
        "messages": [],
        "code_history": [],
        "iteration": 0,
        "next_action": "execute",
        "final_report": "",
        "phase": "eda",
        "pending_proposal": "",
        "user_feedback": "",
        "consecutive_errors": 0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "started_at": "",
        "domain": domain,
        "artifact_contract": contract,
        "verification_result": {},
        "verification_retries": 0,
        "max_verification_retries": max(0, retries),
        "verification_passed": False,
    }
