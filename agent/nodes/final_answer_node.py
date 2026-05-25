from __future__ import annotations

from agent.state import FinancialAgentState
from utils.logger_handler import log_stage


def final_answer_node(state: FinancialAgentState) -> dict:
    with log_stage("final_answer") as stage:
        critique = state.get("critique_result", {})
        final_answer = state.get("final_answer") or state.get("draft_answer", "")
        issues = critique.get("issues", []) if isinstance(critique, dict) else []
        if critique and not critique.get("passed", True):
            final_answer += f"\n\n审查提示：{'；'.join(issues)}"

        reflection_history = state.get("reflection_history", [])
        if reflection_history:
            rounds_info = []
            for entry in reflection_history:
                round_num = entry.get("round", "?")
                evidence_before = entry.get("evidence_count_before", "?")
                evidence_after = entry.get("evidence_count_after", "?")
                issues_text = "；".join(entry.get("critique_issues", [])[:3]) if entry.get("critique_issues") else "无"
                rounds_info.append(f"第{round_num}轮：证据从{evidence_before}条→{evidence_after}条，审查问题：{issues_text}")
            final_answer += "\n\n反思过程：" + " → ".join(rounds_info)

        final_answer = final_answer.strip()
        stage.add_done_fields(final_answer_chars=len(final_answer), critique_issues=len(issues), reflection_rounds=len(reflection_history))
        return {"final_answer": final_answer}