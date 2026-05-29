from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agent.nodes.calculator_node import calculator_node     # 相关节点导入
from agent.nodes.citation_guard_node import citation_guard_node
from agent.nodes.critic_node import critic_node
from agent.nodes.final_answer_node import final_answer_node
from agent.nodes.graph_rag_node import graph_rag_node
from agent.nodes.query_transform_node import query_transform_node
from agent.nodes.reasoning_node import reasoning_node
from agent.nodes.retrieval_node import retrieval_node
from agent.nodes.router_node import router_node
from agent.nodes.web_search_node import web_search_node
from agent.state import FinancialAgentState  # 导入状态、记忆、配置和日志工具    FinancialAgentState是全局状态节点
from memory.heartbeat import MemoryHeartbeat
from memory.short_term import ShortTermMemory
from memory.user_profile import UserProfileService
from utils.config_handler import agent_cof, graph_cof, memory_cof, rag_cof
from utils.helpers import as_bool
from utils.logger_handler import log_stage, safe_preview


class FinancialGraphAgent:
    """Financial multi-hop analysis agent with optional LangGraph runtime."""

    def __init__(self, user_id: str):
        if not user_id:
            raise ValueError("user_id不能为空——请通过登录页传入合法用户名")
        self.user_id = user_id
        self.short_memory = ShortTermMemory(window_size=int(memory_cof.get("short_term_window", 8)))
        self.short_memory.restore(user_id)  # L1：从存储层恢复summary+slots
        self.heartbeat = MemoryHeartbeat()
        self.profiles = UserProfileService()
        self.compiled_graph = self._try_build_langgraph()

    def _try_build_langgraph(self):
        try:
            from langgraph.graph import END, START, StateGraph

            enable_graph_rag = as_bool(rag_cof.get("enable_graph_rag"), default=True)       # 图rag
            enable_web_fallback = as_bool(graph_cof.get("enable_web_fallback"), default=False)      # web搜索后备
            enable_critic = as_bool(graph_cof.get("enable_critic"), default=True)   # 评审机制
            enable_citation_guard = as_bool(graph_cof.get("enable_citation_guard"), default=True)   # 引用检查机制
            max_reflection_rounds = int(graph_cof.get("max_reflection_rounds", 2))  # 最大反思轮数

            workflow = StateGraph(FinancialAgentState)
            workflow.add_node("router", router_node)
            workflow.add_node("query_transform", query_transform_node)      # 感觉这个改写有点鸡肋，很复杂
            workflow.add_node("retrieval", retrieval_node)
            if enable_graph_rag:
                workflow.add_node("graph_rag", graph_rag_node)
            workflow.add_node("calculator", calculator_node)
            workflow.add_node("web_search", web_search_node)
            workflow.add_node("reasoning", reasoning_node)
            if enable_critic:
                workflow.add_node("critic", critic_node)
            if enable_citation_guard:
                workflow.add_node("citation_guard", citation_guard_node)
            workflow.add_node("final_answer", final_answer_node)

            workflow.add_edge(START, "router")
            workflow.add_edge("router", "query_transform")
            workflow.add_edge("query_transform", "retrieval")
            if enable_graph_rag:        # if graph_rag就走这个
                workflow.add_edge("retrieval", "graph_rag")
                workflow.add_edge("graph_rag", "calculator")
            else:
                workflow.add_edge("retrieval", "calculator")

            def route_after_calculator(state: FinancialAgentState) -> str:
                if state.get("needs_web_search") or enable_web_fallback:
                    return "web_search"
                return "reasoning"

            workflow.add_conditional_edges(
                "calculator",
                route_after_calculator,
                {
                    "web_search": "web_search",
                    "reasoning": "reasoning",
                },
            )
            workflow.add_edge("web_search", "reasoning")

            def route_after_reasoning(_: FinancialAgentState) -> str:
                if enable_critic:
                    return "critic"
                if enable_citation_guard:
                    return "citation_guard"
                return "final_answer"

            after_reasoning_edges = {"final_answer": "final_answer"}
            if enable_critic:
                after_reasoning_edges["critic"] = "critic"
            if enable_citation_guard:
                after_reasoning_edges["citation_guard"] = "citation_guard"
            workflow.add_conditional_edges("reasoning", route_after_reasoning, after_reasoning_edges)

            enable_query_rewrite = as_bool(rag_cof.get("enable_query_rewrite"), default=True)

            if enable_critic:
                if enable_query_rewrite:
                    def route_after_critic(state: FinancialAgentState) -> str:\
                        # 1. 看质检报告
                        critique = state.get("critique_result", {})
                        blocking_issues = [i for i in critique.get("issues", []) if i.startswith("BLOCKING:")]

                        # 2. 判断是否需要“打回重做”
                        needs_more = bool(
                            critique.get("needs_more_evidence", False)
                            or (blocking_issues and state.get("reflection_round", 0) < max_reflection_rounds)
                        )

                        # 3. 决定去向
                        if needs_more and state.get("reflection_round", 0) < max_reflection_rounds:
                            return "query_transform"    # # 路线 A：打回重做
                        if enable_citation_guard:
                            return "citation_guard"     # # 路线 B：通过质检，送去加护栏
                        return "final_answer"           # # 路线 C：通过质检，直接打包

                    critic_edges = {"final_answer": "final_answer"}
                    critic_edges["query_transform"] = "query_transform"
                    if enable_citation_guard:
                        critic_edges["citation_guard"] = "citation_guard"
                    workflow.add_conditional_edges("critic", route_after_critic, critic_edges)
                else:
                    def route_after_critic_no_reflection(state: FinancialAgentState) -> str:
                        if enable_citation_guard:
                            return "citation_guard"
                        return "final_answer"

                    critic_edges_no_reflection = {"final_answer": "final_answer"}
                    if enable_citation_guard:
                        critic_edges_no_reflection["citation_guard"] = "citation_guard"
                    workflow.add_conditional_edges("critic", route_after_critic_no_reflection, critic_edges_no_reflection)

            if enable_citation_guard:
                workflow.add_edge("citation_guard", "final_answer")
            workflow.add_edge("final_answer", END)
            return workflow.compile()
        except Exception:
            return None

    def _initial_state(self, query: str) -> FinancialAgentState:
        profile = self.profiles.get(self.user_id)
        return {
            "messages": list(self.short_memory.messages) + [{"role": "user", "content": query}],
            "user_id": self.user_id,
            "user_profile": profile,      # 画像回灌：供下游节点参考
            "user_query": query,
            "reflection_round": 0,
            "reflection_history": [],
            "memory_snapshot": self.short_memory.snapshot(),
        }

    def invoke(self, query: str) -> FinancialAgentState:
        with log_stage("agent.invoke", user_id=self.user_id, query=safe_preview(query)) as stage:
            state = self._initial_state(query)
            if self.compiled_graph is not None:
                state = self.compiled_graph.invoke(state)
            else:
                state = self._invoke_fallback(state)

            final_answer = state.get("final_answer", "")
            self.short_memory.append("user", query)
            self.short_memory.append("assistant", final_answer)
            if self.heartbeat.should_compact(list(self.short_memory.messages)):
                self.short_memory.summary = self.heartbeat.summarize(
                    list(self.short_memory.messages), self.short_memory.summary
                )
            entities = state.get("entities", {})
            self.profiles.remember_focus(
                self.user_id,
                entities.get("companies", []),
                entities.get("metrics", []),
            )
            # ── 画像风格沉淀：简单规则推断，后续可改LLM ──
            draft_len = len(state.get("draft_answer", ""))
            if draft_len > 500:
                inferred_style = "professional"
            elif draft_len > 0:
                inferred_style = "casual"
            else:
                inferred_style = "professional"
            self.profiles.remember_style(
                self.user_id,
                language_style=inferred_style,
                risk_preference="neutral",
            )
            # ── L1：短期记忆持久化（关浏览器不丢失summary+slots）──
            self.short_memory.persist(self.user_id)
            stage.add_done_fields(
                final_answer_chars=len(final_answer),
                evidence_cards=len(state.get("evidence_cards", [])),
                intent=state.get("intent"),
                reflection_round=state.get("reflection_round", 0),
            )
            return state

    def _invoke_fallback(self, state: FinancialAgentState) -> FinancialAgentState:      # 回退/兜底/降级机制
        max_reflection_rounds = int(graph_cof.get("max_reflection_rounds", 2))
        for node in [router_node, query_transform_node, retrieval_node]:
            state.update(node(state))
        if as_bool(rag_cof.get("enable_graph_rag"), default=True):
            state.update(graph_rag_node(state))
        else:
            state["graph_relations"] = []
        state.update(calculator_node(state))
        if state.get("needs_web_search") or as_bool(graph_cof.get("enable_web_fallback"), default=False):
            state.update(web_search_node(state))
        else:
            state.setdefault("web_search_note", "")
        state.update(reasoning_node(state))
        if as_bool(graph_cof.get("enable_critic"), default=True):
            state.update(critic_node(state))
            critique = state.get("critique_result", {})
            enable_query_rewrite = as_bool(rag_cof.get("enable_query_rewrite"), default=True)
            if enable_query_rewrite:
                blocking_issues = [i for i in critique.get("issues", []) if i.startswith("BLOCKING:")]
                needs_more = bool(
                    critique.get("needs_more_evidence", False)
                    or (blocking_issues and state.get("reflection_round", 0) < max_reflection_rounds)
                )
                while needs_more and state.get("reflection_round", 0) < max_reflection_rounds:
                    history_entry = {
                        "round": state.get("reflection_round", 0) + 1,
                        "critique_issues": critique.get("issues", []),
                        "evidence_count_before": len(state.get("evidence_cards", [])),
                    }
                    state["reflection_round"] = state.get("reflection_round", 0) + 1
                    state.update(query_transform_node(state))
                    state.update(retrieval_node(state))
                    if as_bool(rag_cof.get("enable_graph_rag"), default=True):
                        state.update(graph_rag_node(state))
                    state.update(calculator_node(state))
                    if state.get("needs_web_search") or as_bool(graph_cof.get("enable_web_fallback"), default=False):
                        state.update(web_search_node(state))
                    state.update(reasoning_node(state))
                    state.update(critic_node(state))
                    critique = state.get("critique_result", {})
                    history_entry["evidence_count_after"] = len(state.get("evidence_cards", []))
                    history_entry["critique_passed_after"] = critique.get("passed", False)
                    state.setdefault("reflection_history", []).append(history_entry)
                    blocking_issues = [i for i in critique.get("issues", []) if i.startswith("BLOCKING:")]
                    needs_more = bool(
                        critique.get("needs_more_evidence", False)
                        or (blocking_issues and state.get("reflection_round", 0) < max_reflection_rounds)
                    )
        if as_bool(graph_cof.get("enable_citation_guard"), default=True):
            state.update(citation_guard_node(state))
        state.update(final_answer_node(state))
        return state

    def invoke_stream(self, query: str) -> tuple[FinancialAgentState, str]:
        """流式模式：管线前置步骤同步完成，reasoning只构建prompt不调LLM。
        返回 (state, reasoning_prompt)，由调用方负责stream LLM渲染。
        如果evidence_cards为空，reasoning_prompt为空字符串，直接用fallback draft_answer。"""
        state = self._initial_state(query)
        state["skip_llm_reasoning"] = True  # 告诉reasoning_node跳过LLM
        if self.compiled_graph is not None:
            state = self.compiled_graph.invoke(state)
        else:
            state = self._invoke_fallback(state)

        reasoning_prompt = state.get("reasoning_prompt", "")
        # 如果无证据→reasoning_node走fallback，draft_answer已有内容，无需流式
        if not reasoning_prompt and state.get("draft_answer"):
            # fallback路径已有完整答案，直接包装为final_answer
            state["final_answer"] = state["draft_answer"]
            return state, ""  # 空 prompt = 不需要流式

        # 后处理（记忆、画像）在_stream_finalize中完成
        return state, reasoning_prompt

    def _stream_finalize(self, state: FinancialAgentState, answer: str) -> None:
        """流式完成后：更新记忆、画像、日志——与invoke()的后处理逻辑一致。"""
        state["final_answer"] = answer
        state["draft_answer"] = answer
        self.short_memory.append("assistant", answer)
        if self.heartbeat.should_compact(list(self.short_memory.messages)):
            self.short_memory.summary = self.heartbeat.summarize(
                list(self.short_memory.messages), self.short_memory.summary
            )
        entities = state.get("entities", {})
        self.profiles.remember_focus(
            self.user_id,
            entities.get("companies", []),
            entities.get("metrics", []),
        )
        draft_len = len(answer)
        inferred_style = "professional" if draft_len > 500 else ("casual" if draft_len > 0 else "professional")
        self.profiles.remember_style(self.user_id, language_style=inferred_style, risk_preference="neutral")
        # ── L1：短期记忆持久化 ──
        self.short_memory.persist(self.user_id)


if __name__ == "__main__":
    agent = FinancialGraphAgent(user_id="test_user")
    result = agent.invoke("示例科技2024年现金流质量如何？")
    print(result["final_answer"])