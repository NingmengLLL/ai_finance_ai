from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

from agent.graph import FinancialGraphAgent
from memory import ConversationStore, sanitize_user_id
import warnings
import transformers
import warnings

def run_cli() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="cli_user", help="用户ID（默认cli_user）")
    parser.add_argument("query", nargs="*", help="查询文本")
    args = parser.parse_args()
    query = " ".join(args.query).strip() or "示例科技2024年现金流质量如何？"
    agent = FinancialGraphAgent(user_id=args.user)
    result = agent.invoke(query)
    print(result["final_answer"])


def _now_label() -> str:
    return datetime.now().strftime("%H:%M")


def _thread_title(query: str) -> str:
    compact = " ".join(query.split())
    return compact if len(compact) <= 28 else compact[:28].rstrip() + "..."


def _make_thread(thread_id: str) -> dict[str, Any]:
    return {
        "id": thread_id,
        "title": "新对话",
        "messages": [],
        "last_state": {},
        "updated_at": "",
    }


def render_login_page() -> str | None:
    """登录/选择用户页面。返回选定的user_id或None。"""
    import streamlit as st

    conv_store = st.session_state.get("conversation_store")
    existing_users = conv_store.list_users() if conv_store else []

    st.markdown("""
        <div class="hero-wrap" style="margin-top:6vh;text-align:center;">
            <div class="hero-kicker">欢迎回来</div>
            <h1 class="hero-title">金融研报分析助手</h1>
            <div class="hero-body">选择或输入用户名以加载您的对话历史与偏好画像。</div>
        </div>
        """, unsafe_allow_html=True)

    _, center, _ = st.columns([0.2, 0.6, 0.2])
    with center:
        st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
        user_id = st.text_input("输入新用户名", placeholder="如 analyst_zhang", key="login-input")
        if existing_users:
            selected = st.selectbox("或选择已有用户", existing_users, key="login-select")
            if selected and not user_id:
                user_id = selected

        if st.button("进入工作台", use_container_width=True, key="login-btn"):
            if user_id and user_id.strip():
                try:
                    return sanitize_user_id(user_id.strip())
                except ValueError as e:
                    st.error(str(e))
        st.markdown('</div>', unsafe_allow_html=True)
    return None

def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="金融研报分析助手",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with main_col:
        st.markdown(
            """
            <div class="topbar">
                <div class="brand-mark"></div>
                <div class="topbar-copy">金融研究工作台</div>
            </div>
            """,
            unsafe_allow_html=True,
        )