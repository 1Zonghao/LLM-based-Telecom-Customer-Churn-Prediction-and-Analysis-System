import re
from datetime import datetime, timezone
from typing import Callable

from .state import AgentState
from .llm_client import LLMClient
from .prompts import (
    build_system_prompt,
    TASK_PROMPT,
    CONTINUE_PROMPT,
    ERROR_CONTINUE_PROMPT,
    VERIFICATION_CONTINUE_PROMPT,
)
from .verifier import (
    format_verification_feedback,
    format_verification_report,
    verify_artifacts,
)
from sandbox.kernel_manager import KernelManager
from sandbox.compressor import OutputCompressor

CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
CHECKPOINT_FE_RE = re.compile(r"\[CHECKPOINT:FE\]")
CHECKPOINT_MODEL_RE = re.compile(r"\[CHECKPOINT:MODEL\]")
MAX_ITERATIONS = 10


def make_reasoning_node(llm: LLMClient) -> Callable:
    """创建 reasoning_node：调用 LLM 推理并决定下一步动作（含断点检测）。"""

    def reasoning_node(state: AgentState) -> dict:
        # 安全阀：达到迭代上限后不再调用 LLM，直接进入交付验证。
        # 每次验证重试额外给 2 次动作预算，避免“修复代码还没跑就被截断”。
        max_iterations = MAX_ITERATIONS + state.get("verification_retries", 0) * 2
        if state.get("iteration", 0) >= max_iterations:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "[达到最大迭代次数，进入交付验证]",
                }],
                "final_report": state.get("final_report", "") or "[达到最大迭代次数，自动终止]",
                "next_action": "finish",
                "phase": "reporting",
            }

        messages = _build_messages(state)
        response, usage = llm.chat(messages)

        # 累加 token 消耗
        prev_usage = state.get("token_usage", {})
        new_usage = {
            "prompt_tokens": prev_usage.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0),
            "completion_tokens": prev_usage.get("completion_tokens", 0) + usage.get("completion_tokens", 0),
            "total_tokens": prev_usage.get("total_tokens", 0) + usage.get("total_tokens", 0),
        }

        # CHECKPOINT 标记优先于代码块：LLM 可能在提案中附带示例代码
        is_fe_checkpoint = CHECKPOINT_FE_RE.search(response)
        is_model_checkpoint = CHECKPOINT_MODEL_RE.search(response)
        code_match = CODE_BLOCK_RE.search(response)

        updates: dict = {
            "messages": [{"role": "assistant", "content": response}],
            "token_usage": new_usage,
        }

        # 首次调用时记录开始时间
        if not state.get("started_at"):
            updates["started_at"] = datetime.now(timezone.utc).isoformat()

        if is_fe_checkpoint:
            proposal = _extract_proposal(response)
            updates["pending_proposal"] = proposal
            updates["next_action"] = "checkpoint_fe"

        elif is_model_checkpoint:
            proposal = _extract_proposal(response)
            updates["pending_proposal"] = proposal
            updates["next_action"] = "checkpoint_model"

        elif code_match:
            code = code_match.group(1).strip()
            updates["code_history"] = [code]
            updates["next_action"] = "execute"

        else:
            updates["final_report"] = response
            updates["next_action"] = "finish"
            updates["phase"] = "reporting"

        return updates

    return reasoning_node


def make_action_node(km: KernelManager, compressor: OutputCompressor) -> Callable:
    """创建 action_node：在沙箱中执行代码并压缩结果。"""

    def action_node(state: AgentState) -> dict:
        code_history = state.get("code_history", [])
        if not code_history:
            return {
                "messages": [{"role": "system", "content": "[无代码可执行]"}],
                "iteration": state["iteration"] + 1,
            }

        code = code_history[-1]
        result = km.execute(code)
        observation = compressor.compress(result)
        profile_update = _extract_profile_from_output(result.text, state["data_profile"])

        # action 节点只负责执行代码和压缩结果，不推断阶段。
        # phase 的推进必须由 checkpoint 节点（用户确认后）或 finish 显式控制。

        # 错误追踪
        consecutive_errors = state.get("consecutive_errors", 0)
        if result.has_error:
            consecutive_errors += 1
            # 附带代码原文，方便 LLM 对照错误修改
            msg_content = f"[执行错误]\n{observation}\n\n[出错代码]\n```python\n{code}\n```"
        else:
            consecutive_errors = 0
            msg_content = f"[执行结果]\n{observation}"

        updates = {
            "messages": [{"role": "user", "content": msg_content}],
            "data_profile": {**state["data_profile"], **profile_update},
            "iteration": state["iteration"] + 1,
            "consecutive_errors": consecutive_errors,
        }

        return updates

    return action_node


def make_checkpoint_fe_node() -> Callable:
    """特征工程检查点节点：由 graph.compile(interrupt_before=...) 控制挂起。

    当用户确认后，此节点被实际执行，负责推进阶段并清空 pending_proposal。
    若用户提供了反馈，将其注入 messages 供后续 reasoning 节点读取。
    """

    def checkpoint_fe_node(state: AgentState) -> dict:
        feedback = state.get("user_feedback", "")
        updates = {
            "phase": "feature_engineering",
            "next_action": "execute",
            "pending_proposal": "",
            "user_feedback": "",
        }
        if feedback.strip():
            updates["messages"] = [{
                "role": "user",
                "content": f"【用户反馈】{feedback.strip()}"
            }]
        return updates

    return checkpoint_fe_node


def make_checkpoint_model_node() -> Callable:
    """模型训练检查点节点：由 graph.compile(interrupt_before=...) 控制挂起。

    当用户确认后，此节点被实际执行，负责推进阶段并清空 pending_proposal。
    若用户提供了反馈，将其注入 messages 供后续 reasoning 节点读取。
    """

    def checkpoint_model_node(state: AgentState) -> dict:
        feedback = state.get("user_feedback", "")
        updates = {
            "phase": "model_training",
            "next_action": "execute",
            "pending_proposal": "",
            "user_feedback": "",
        }
        if feedback.strip():
            updates["messages"] = [{
                "role": "user",
                "content": f"【用户反馈】{feedback.strip()}"
            }]
        return updates

    return checkpoint_model_node


def make_verifier_node() -> Callable:
    """交付验证节点：检查契约，决定通过、修复还是终止。"""

    def verifier_node(state: AgentState) -> dict:
        contract = state.get("artifact_contract") or {}
        result = verify_artifacts(contract, state)

        if result["passed"]:
            updates: dict = {
                "verification_result": result,
                "verification_history": [result],
                "verification_passed": True,
                "next_action": "finish",
                "phase": "completed",
                "messages": [{
                    "role": "system",
                    "content": "[验证通过] 交付物检查全部通过。",
                }],
            }
            if not state.get("final_report"):
                lines = ["### 交付验证通过", ""]
                for check in result.get("checks", []):
                    lines.append(f"- PASS `{check['name']}` — {check.get('detail', '')}")
                updates["final_report"] = "\n".join(lines)
            return updates

        retries = int(state.get("verification_retries", 0))
        max_retries = int(state.get("max_verification_retries", 2))

        if retries >= max_retries:
            return {
                "verification_result": result,
                "verification_history": [result],
                "verification_passed": False,
                "next_action": "finish",
                "phase": "verification_failed",
                "final_report": format_verification_report(result),
                "messages": [{
                    "role": "system",
                    "content": "[验证失败] 已达到最大重试次数，流程终止。",
                }],
            }

        return {
            "verification_result": result,
            "verification_history": [result],
            "verification_passed": False,
            "verification_retries": retries + 1,
            "next_action": "repair",
            "phase": "verification_repair",
            "messages": [{
                "role": "user",
                "content": format_verification_feedback(result, contract),
            }],
        }

    return verifier_node


def verifier_router(state: AgentState) -> str:
    """验证节点路由：需要修复则回到 reasoning，否则结束。"""
    return "repair" if state.get("next_action") == "repair" else "finish"


def router(state: AgentState) -> str:
    """条件路由：根据 next_action 决定分支。"""
    action = state.get("next_action", "finish")
    valid = {"execute", "checkpoint_fe", "checkpoint_model", "finish"}
    if action not in valid:
        return "finish"
    return action


def _build_messages(state: AgentState) -> list[dict]:
    """构建发送给 LLM 的消息列表。"""
    profile = state.get("data_profile", {})
    profile_str = _format_profile(profile)

    system_prompt = build_system_prompt(
        state.get("domain", "generic"),
        profile_str,
    )

    msgs: list[dict] = [{"role": "system", "content": system_prompt}]

    if not state.get("messages"):
        msgs.append({
            "role": "user",
            "content": TASK_PROMPT.format(
                task=state["task"],
                data_path=state["data_path"],
                phase=state.get("phase", "eda"),
                artifact_contract=_format_contract(state.get("artifact_contract")),
                skill_context=state.get("skill_context") or "暂无",
            ),
        })
    else:
        for m in state["messages"]:
            msgs.append(m)

        # 检测最后一条消息是执行错误还是交付验证失败
        last_msg = state["messages"][-1] if state["messages"] else {}
        last_content = last_msg.get("content", "")
        is_error = "[执行错误]" in last_content
        is_verification_failure = "[验证失败]" in last_content

        if is_verification_failure:
            msgs.append({
                "role": "user",
                "content": VERIFICATION_CONTINUE_PROMPT.format(observation=last_content),
            })
        elif is_error:
            # 提取错误信息和代码
            error_text = last_content
            last_code = state.get("code_history", [])[-1] if state.get("code_history") else ""
            consecutive = state.get("consecutive_errors", 0)
            msgs.append({
                "role": "user",
                "content": ERROR_CONTINUE_PROMPT.format(
                    observation=error_text,
                    code=last_code,
                    phase=state.get("phase", "eda"),
                    consecutive_errors=consecutive,
                ),
            })
        else:
            msgs.append({
                "role": "user",
                "content": CONTINUE_PROMPT.format(
                    observation="（请根据以上执行结果继续分析）",
                    phase=state.get("phase", "eda"),
                ),
            })

    return msgs


def _format_contract(contract: dict | None) -> str:
    """把交付物契约格式化成 LLM 可读文本。"""
    if not contract or not contract.get("required_files"):
        return "无（按任务描述执行）"

    lines: list[str] = []
    for spec in contract.get("required_files", []):
        path = spec.get("path", "")
        fmt = spec.get("format", "")
        details = []
        if spec.get("columns"):
            details.append(f"列={spec['columns']}")
        if spec.get("expected_rows") is not None:
            details.append(f"行数={spec['expected_rows']}")
        suffix = f"（{', '.join(details)}）" if details else ""
        lines.append(f"- {path} [{fmt}]{suffix}")
    if contract.get("labels_path"):
        lines.append(f"- 标签文件：{contract['labels_path']}")
    if contract.get("metric"):
        lines.append(f"- 评估指标：{contract['metric']}")
    if contract.get("group_column"):
        lines.append(f"- 分组列：{contract['group_column']}（必须使用分组交叉验证）")
    return "\n".join(lines)


def _extract_proposal(response: str) -> str:
    """从 LLM 响应中提取方案文本（去除 CHECKPOINT 标记本身）。"""
    text = CHECKPOINT_FE_RE.sub("", response)
    text = CHECKPOINT_MODEL_RE.sub("", text)
    return text.strip()


def _format_profile(profile: dict) -> str:
    """将 data_profile 格式化为 LLM 可读文本。"""
    if not profile:
        return "暂无已知数据信息。"

    lines = ["已知数据信息："]
    key_labels = {
        "n_rows": "样本数",
        "n_cols": "特征数",
        "best_f1": "最佳 F1",
        "best_auc": "最佳 AUC",
        "best_recall": "最佳 Recall",
    }
    for key, value in profile.items():
        label = key_labels.get(key, key)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _extract_profile_from_output(text: str, current: dict) -> dict:
    """从执行输出中提取关键指标，更新 data_profile。"""
    updates = {}

    shape_match = re.search(r"\((\d+),\s*(\d+)\)", text)
    if shape_match:
        updates["n_rows"] = int(shape_match.group(1))
        updates["n_cols"] = int(shape_match.group(2))

    f1_patterns = [
        r"F1[-\s]?[Ss]core[:\s]*([\d.]+)",
        r"f1-score[:\s]*([\d.]+)",
        r"'f1'[:\s]*([\d.]+)",
        r"F1[:\s]+([\d.]+)",
    ]
    for pat in f1_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            score = float(m.group(1))
            prev_best = max(current.get("best_f1", 0), updates.get("best_f1", 0))
            if score > prev_best:
                updates["best_f1"] = score
            break

    auc_patterns = [
        r"AUC[:\s]+([\d.]+)",
        r"roc_auc[:\s]+([\d.]+)",
        r"ROC AUC[:\s]+([\d.]+)",
    ]
    for pat in auc_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            score = float(m.group(1))
            prev_best = max(current.get("best_auc", 0), updates.get("best_auc", 0))
            if score > prev_best:
                updates["best_auc"] = score
            break

    recall_patterns = [
        r"[Rr]ecall[:\s]+([\d.]+)",
        r"'recall'[:\s]+([\d.]+)",
    ]
    for pat in recall_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            score = float(m.group(1))
            prev_best = max(current.get("best_recall", 0), updates.get("best_recall", 0))
            if score > prev_best:
                updates["best_recall"] = score
            break

    return updates


def _is_feature_engineering_code(code: str) -> bool:
    """推断代码是否属于特征工程阶段。"""
    fe_keywords = [
        "fillna", "dropna", "impute", "SimpleImputer",
        "get_dummies", "OneHotEncoder", "LabelEncoder", "OrdinalEncoder",
        "StandardScaler", "MinMaxScaler", "scale", "normalize",
        "SMOTE", "smote", "train_test_split",
        "replace(", "map(", "apply(",
        "astype(", "to_numeric",
    ]
    return any(kw in code for kw in fe_keywords)


def _is_model_training_code(code: str) -> bool:
    """推断代码是否属于模型训练阶段。"""
    model_keywords = [
        "XGBClassifier", "XGBRegressor",
        "RandomForestClassifier", "RandomForestRegressor",
        "LogisticRegression",
        "SVC", "SVR",
        "GradientBoostingClassifier",
        "fit(", "predict(", "predict_proba(",
        "cross_val_score", "cross_validate", "StratifiedKFold",
        "GridSearchCV", "RandomizedSearchCV",
        "classification_report", "confusion_matrix",
        "roc_curve", "roc_auc_score", "f1_score", "recall_score",
    ]
    return any(kw in code for kw in model_keywords)
