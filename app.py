from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

from agent.graph import FinancialGraphAgent
from memory import ConversationStore, sanitize_user_id
import warnings
import transformers
import warnings

# 忽略所有乱七八糟的 Python 警告
warnings.filterwarnings("ignore")

import jieba
# ... 其他代码
# 将日志级别设置为 ERROR，这样就不会打印 Warning 了
transformers.logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

APP_NAME = "金融研报分析助手"
APP_SUBTITLE = "提出问题、继续本地研究，或开启一段新的分析线程。"
RECOMMENDED_QUERIES = [
    "腾讯2024年现金流质量如何？",
    "美团2024年收入增长怎么看？",
    "比较腾讯和美团的盈利质量",
]


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


def _load_user_conversations(user_id: str) -> None:
    """从持久化存储加载该用户的对话历史到session_state。"""
    import streamlit as st

    conv_store = st.session_state["conversation_store"]
    threads = conv_store.get_threads(user_id)
    if threads:
        st.session_state["threads"] = {t["id"]: t for t in threads}
        st.session_state["thread_order"] = [t["id"] for t in threads]
        st.session_state["active_thread_id"] = threads[0]["id"] if threads else "thread-1"
        # ── 关键：thread_counter必须跳过已有ID，避免撞key ──
        existing_nums = []
        for t in threads:
            tid = t["id"]
            if tid.startswith("thread-"):
                try:
                    existing_nums.append(int(tid.split("-", 1)[1]))
                except ValueError:
                    pass
        max_num = max(existing_nums) if existing_nums else 0
        st.session_state["thread_counter"] = max_num + 1
    else:
        thread_id = f"thread-{st.session_state.get('thread_counter', 1)}"
        st.session_state["threads"] = {thread_id: _make_thread(thread_id)}
        st.session_state["thread_order"] = [thread_id]
        st.session_state["active_thread_id"] = thread_id


def inject_chat_css() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f7f3ec;
            --panel-bg: rgba(255, 255, 255, 0.86);
            --panel-strong: #ffffff;
            --border: #e7dece;
            --text-main: #2f2b25;
            --text-soft: #8d8376;
            --accent: #c9843d;
            --accent-soft: #f4e1cc;
            --shadow: 0 16px 40px rgba(67, 50, 30, 0.08);
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background:
                radial-gradient(circle at top right, rgba(237, 214, 189, 0.58), transparent 30%),
                linear-gradient(180deg, #fbfaf7 0%, var(--app-bg) 100%);
            color: var(--text-main);
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            box-shadow: none;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        section[data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 18rem;
            width: 18rem !important;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.74);
            border-right: 1px solid rgba(231, 222, 206, 0.9);
            backdrop-filter: blur(18px);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.2rem;
        }

        [data-testid="stSidebar"] button {
            border-radius: 14px;
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--border);
            color: var(--text-main);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(180deg, #fff6ea 0%, #f4e1cc 100%);
            border: 1px solid rgba(201, 132, 61, 0.28);
            color: #7b552f;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin-bottom: 1rem;
        }

        .brand-mark {
            width: 1.85rem;
            height: 1.85rem;
            border-radius: 0.65rem;
            background: linear-gradient(145deg, #e3a361 0%, #c9843d 100%);
            box-shadow: 0 12px 24px rgba(201, 132, 61, 0.22);
        }

        .brand-copy {
            display: flex;
            flex-direction: column;
            gap: 0.05rem;
        }

        .brand-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .brand-meta {
            font-size: 0.74rem;
            color: var(--text-soft);
        }

        .section-label {
            margin: 1rem 0 0.5rem;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--text-soft);
        }

        .recent-meta,
        .sidebar-empty {
            font-size: 0.78rem;
            color: var(--text-soft);
            margin: 0.15rem 0 0.55rem 0.1rem;
        }

        .sidebar-footer {
            margin-top: 1.4rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(231, 222, 206, 0.85);
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .footer-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(231, 222, 206, 0.95);
            color: var(--text-soft);
            font-size: 0.8rem;
        }

        .footer-dot {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 999px;
            background: #63ba7e;
        }

        .topbar {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.65rem;
            padding: 0.2rem 0 0.8rem;
        }

        .topbar .brand-mark {
            width: 1.1rem;
            height: 1.1rem;
            border-radius: 0.4rem;
            box-shadow: none;
        }

        .topbar-copy {
            font-size: 0.88rem;
            color: var(--text-soft);
            font-weight: 600;
        }

        .hero-wrap {
            max-width: 760px;
            margin: 8vh auto 0;
            text-align: left;
        }

        .hero-kicker {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.55rem;
        }

        .hero-title {
            font-size: clamp(2rem, 4vw, 3.35rem);
            line-height: 1.02;
            letter-spacing: 0;
            margin: 0;
            color: var(--text-main);
        }

        .hero-body {
            max-width: 580px;
            margin-top: 0.8rem;
            font-size: 1rem;
            line-height: 1.7;
            color: var(--text-soft);
        }

        div[data-testid="stForm"] {
            background: var(--panel-bg);
            border: 1px solid rgba(231, 222, 206, 0.95);
            border-radius: 24px;
            box-shadow: var(--shadow);
            margin-top: 1.45rem;
            padding: 0.8rem 0.85rem 0.65rem;
        }

        div[data-testid="stTextArea"] textarea {
            background: transparent;
            color: var(--text-main);
            border: none;
            box-shadow: none;
            font-size: 1rem;
        }

        div[data-testid="stTextArea"] textarea::placeholder {
            color: #9a9185;
        }

        .composer-note {
            font-size: 0.8rem;
            color: var(--text-soft);
            margin: 0.35rem 0 0.2rem;
        }

        div[data-testid="stFormSubmitButton"] button {
            border-radius: 999px;
            background: linear-gradient(180deg, #dba163 0%, #c9843d 100%);
            color: #fffdf8;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 700;
        }

        .suggestions-row {
            margin-top: 1rem;
        }

        div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(231, 222, 206, 0.95);
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 0.25rem 0.45rem;
            margin-bottom: 1rem;
        }

        .main-shell [data-testid="stChatMessageContent"] p,
        .main-shell [data-testid="stChatMessageContent"] li {
            line-height: 1.7;
        }

        .panel-shell {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(231, 222, 206, 0.95);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 0.9rem;
        }

        .panel-title {
            font-size: 0.88rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-soft);
            margin-bottom: 0.85rem;
        }

        .status-card {
            border-radius: 18px;
            padding: 0.9rem 0.95rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(231, 222, 206, 0.95);
            background: var(--panel-strong);
        }

        .status-card strong {
            display: block;
            margin-bottom: 0.2rem;
            color: var(--text-main);
        }

        .status-card p {
            margin: 0;
            color: var(--text-soft);
            line-height: 1.6;
            font-size: 0.88rem;
        }

        .status-pass {
            background: linear-gradient(180deg, #f9fff9 0%, #eef8f0 100%);
        }

        .status-warn {
            background: linear-gradient(180deg, #fffaf2 0%, #fdf2e2 100%);
        }

        .status-error {
            background: linear-gradient(180deg, #fff7f6 0%, #fdeaea 100%);
        }

        .evidence-meta {
            font-size: 0.78rem;
            color: var(--text-soft);
            line-height: 1.55;
        }

        .empty-panel {
            border: 1px dashed rgba(201, 132, 61, 0.28);
            border-radius: 18px;
            padding: 1rem;
            color: var(--text-soft);
            line-height: 1.65;
            background: rgba(255, 255, 255, 0.62);
        }

        [data-testid="stChatInput"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(231, 222, 206, 0.95);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 0.15rem 0.3rem;
        }

        @media (max-width: 1100px) {
            .hero-wrap {
                margin-top: 2vh;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    import streamlit as st

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None      # 登录前为None
    if "agent" not in st.session_state:
        pass                                     # 登录后才创建
    if "conversation_store" not in st.session_state:
        st.session_state["conversation_store"] = ConversationStore()
    if "thread_counter" not in st.session_state:
        st.session_state["thread_counter"] = 1
    # ── 已登录但threads不存在 → 从持久化存储加载 ──
    if "threads" not in st.session_state and st.session_state.get("user_id"):
        _load_user_conversations(st.session_state["user_id"])
        return
    # ── 未登录且threads不存在 → 创建初始空线程 ──
    if "threads" not in st.session_state:
        thread_id = f"thread-{st.session_state['thread_counter']}"
        st.session_state["threads"] = {thread_id: _make_thread(thread_id)}
        st.session_state["thread_order"] = [thread_id]
        st.session_state["active_thread_id"] = thread_id
        st.session_state["thread_counter"] += 1


def _active_thread() -> dict[str, Any]:
    import streamlit as st

    return st.session_state["threads"][st.session_state["active_thread_id"]]


def _promote_thread(thread_id: str) -> None:
    """将指定线程提到 thread_order 最前面（最近活跃排序）。"""
    import streamlit as st

    order = st.session_state["thread_order"]
    if thread_id in order:
        order.remove(thread_id)
    order.insert(0, thread_id)


def _create_thread(set_active: bool = True) -> str:
    import streamlit as st

    thread_id = f"thread-{st.session_state['thread_counter']}"
    st.session_state["thread_counter"] += 1
    st.session_state["threads"][thread_id] = _make_thread(thread_id)
    _promote_thread(thread_id)
    if set_active:
        st.session_state["active_thread_id"] = thread_id
    return thread_id


def _persist_threads() -> None:
    """将当前所有对话线程持久化到磁盘/Redis。按 thread_order 顺序写入，保证下次加载排序一致。"""
    import streamlit as st

    conv_store = st.session_state["conversation_store"]
    ordered = [
        st.session_state["threads"][tid]
        for tid in st.session_state["thread_order"]
        if tid in st.session_state["threads"]
    ]
    conv_store.save_threads(st.session_state["user_id"], ordered)


def _start_new_chat() -> None:
    import streamlit as st

    # 先持久化当前对话，防止rerun丢失
    _persist_threads()
    thread = _active_thread()
    if thread["messages"]:
        _create_thread(set_active=True)
    else:
        thread["last_state"] = {}


def _switch_thread(thread_id: str) -> None:
    import streamlit as st

    _persist_threads()
    if thread_id in st.session_state["threads"]:
        st.session_state["active_thread_id"] = thread_id
        _promote_thread(thread_id)


def render_sidebar() -> tuple[str, str | None]:
    import streamlit as st

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="brand-mark"></div>
                <div class="brand-copy">
                    <div class="brand-title">{APP_NAME}</div>
                    <div class="brand-meta">本地研究工作台</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("+ 新对话", use_container_width=True, key="new-chat"):
            return ("new", None)

        st.markdown('<div class="section-label">最近记录</div>', unsafe_allow_html=True)
        threads = st.session_state["threads"]
        active_id = st.session_state["active_thread_id"]
        visible_threads = [
            threads[thread_id]
            for thread_id in st.session_state["thread_order"]
            if threads[thread_id]["messages"] or thread_id == active_id
        ]
        if not visible_threads:
            st.markdown('<div class="sidebar-empty">暂无会话记录。</div>', unsafe_allow_html=True)
        else:
            for thread in visible_threads[:8]:
                is_active = thread["id"] == active_id
                button_type = "primary" if is_active else "secondary"
                if st.button(
                    thread["title"],
                    key=f"thread-{thread['id']}",
                    use_container_width=True,
                    type=button_type,
                ):
                    return ("switch", thread["id"])
                updated_text = f"已更新 {thread['updated_at']}" if thread["updated_at"] else "就绪"
                st.markdown(f'<div class="recent-meta">{updated_text}</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="sidebar-footer">
                <div class="footer-pill"><span class="footer-dot"></span>已连接</div>
                <div class="footer-pill">中文 / 英文</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        user_id = st.session_state.get("user_id", "")
        st.markdown(f'<div class="recent-meta">当前用户：{user_id}</div>', unsafe_allow_html=True)
        if st.button("切换用户", key="switch-user"):
            return ("logout", None)
    return ("stay", None)


def render_empty_state() -> str | None:
    import streamlit as st

    left, center, right = st.columns([0.14, 0.72, 0.14])
    with center:
        st.markdown(
            f"""
            <div class="hero-wrap">
                <div class="hero-kicker">研究工作台</div>
                <h1 class="hero-title">{APP_NAME}</h1>
                <div class="hero-body">{APP_SUBTITLE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("hero-composer", clear_on_submit=True):
            query = st.text_area(
                "提示词",
                height=140,
                placeholder="想了解点什么？",
                label_visibility="collapsed",
            )
            st.markdown(
                '<div class="composer-note">按 Enter 发送，Shift+Enter 换行。</div>',
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button("发送")
        st.markdown('<div class="suggestions-row"></div>', unsafe_allow_html=True)
        suggestion_cols = st.columns(len(RECOMMENDED_QUERIES))
        for index, suggestion in enumerate(RECOMMENDED_QUERIES):
            with suggestion_cols[index]:
                if st.button(suggestion, use_container_width=True, key=f"suggestion-{index}"):
                    return suggestion
        if submitted and query.strip():
            return query.strip()
    return None


def render_messages(messages: list[dict[str, str]]) -> None:
    import streamlit as st

    st.markdown('<div class="main-shell">', unsafe_allow_html=True)
    for message in messages:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
    st.markdown("</div>", unsafe_allow_html=True)


def _critique_summary(state: dict[str, Any]) -> tuple[str, str, str]:
    critique = state.get("critique_result") or {}
    issues = critique.get("issues", []) if isinstance(critique, dict) else []
    if state.get("error"):
        return ("status-error", "执行异常", str(state["error"]))
    if critique.get("passed"):
        return ("status-pass", "审查通过", "证据、引用及回答格式已通过审计检查。")
    if critique.get("needs_more_evidence"):
        return ("status-warn", "需要更多证据", "已生成回答，但审计系统建议提供更广泛的证据支持。")
    if issues:
        return ("status-warn", "审查意见", "；".join(str(item) for item in issues[:3]))
    return ("status-card", "暂无审计结果", "提出问题以查看证据卡片和审计输出。")


def render_evidence_panel(state: dict[str, Any]) -> None:
    import streamlit as st

    st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">证据与审计</div>', unsafe_allow_html=True)

    css_class, heading, body = _critique_summary(state)
    st.markdown(
        f"""
        <div class="status-card {css_class}">
            <strong>{heading}</strong>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = state.get("evidence_cards", []) if isinstance(state, dict) else []
    if not cards:
        st.markdown(
            """
            <div class="empty-panel">
                运行后，证据卡片将显示在此处。此面板在保持回答可读性的同时，会在您需要时展示引用、文本块及审计记录。
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for index, card in enumerate(cards[:6], start=1):
            title = card.get("claim", f"证据 {index}")[:72]
            with st.expander(title, expanded=index == 1):
                st.write(card.get("evidence", ""))
                st.markdown(
                    f"""
                    <div class="evidence-meta">
                        来源：{card.get("source_file", "未知")}<br>
                        页码：{card.get("page_number", "无")}<br>
                        相关度得分：{float(card.get("score", 0.0)):.3f}<br>
                        文本块：{card.get("chunk_id", "无")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    critique = state.get("critique_result")
    if critique:
        with st.expander("审查详情", expanded=False):
            st.json(critique)

    with st.expander("调试状态", expanded=False):
        st.json(state)

    st.markdown("</div>", unsafe_allow_html=True)


def handle_query(query: str) -> None:
    import streamlit as st

    from agent.llm_utils import invoke_llm_stream

    clean_query = query.strip()
    if not clean_query:
        return

    thread = _active_thread()

    # ── 立即渲染用户消息（问完即见，不等答案） ──
    with st.chat_message("user"):
        st.markdown(clean_query)

    thread["messages"].append({"role": "user", "content": clean_query})
    if thread["title"] == "新对话":
        thread["title"] = _thread_title(clean_query)
    thread["updated_at"] = _now_label()

    # ── 进入 assistant 上下文，先显示思考占位 ──
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("⏳ 正在检索财报、重排证据并运行审查...")

        try:
            state, reasoning_prompt = st.session_state["agent"].invoke_stream(clean_query)
        except Exception as exc:
            state = {
                "error": str(exc),
                "evidence_cards": [],
                "critique_result": {
                    "passed": False,
                    "needs_more_evidence": False,
                    "issues": [str(exc)],
                },
            }
            reasoning_prompt = ""

        answer = state.get("final_answer", "")

        if reasoning_prompt:
            # ── 有prompt → 清除思考占位，流式输出 ──
            thinking_placeholder.empty()
            answer = st.write_stream(invoke_llm_stream(reasoning_prompt))
            st.session_state["agent"]._stream_finalize(state, answer)
        elif answer:
            # ── fallback路径：answer已有完整内容，替换占位 ──
            thinking_placeholder.markdown(answer)
            st.session_state["agent"]._stream_finalize(state, answer)
        elif state.get("error"):
            # ── 异常路径 ──
            answer = "运行失败：" + str(state["error"])
            thinking_placeholder.markdown(answer)
        else:
            # ── 兜底 ──
            answer = "系统处理异常，未能生成回答。"
            thinking_placeholder.markdown(answer)

    thread["messages"].append({"role": "assistant", "content": answer})
    thread["last_state"] = state

    # ── 活跃提权 + 持久化 ──
    _promote_thread(st.session_state["active_thread_id"])
    _persist_threads()
    thread["updated_at"] = _now_label()


def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="金融研报分析助手",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_chat_css()
    init_session_state()

    # ── 未登录 → 显示登录页 ──
    if not st.session_state.get("user_id"):
        user_id = render_login_page()
        if user_id:
            st.session_state["user_id"] = user_id
            st.session_state["agent"] = FinancialGraphAgent(user_id=user_id)
            _load_user_conversations(user_id)
            st.rerun()
        return

    # ── 已登录 → 主界面 ──
    sidebar_action, thread_id = render_sidebar()

    # 切换用户 → 回到登录页
    if sidebar_action == "logout":
        st.session_state["user_id"] = None
        st.session_state.pop("agent", None)
        st.rerun()

    if sidebar_action == "new":
        _start_new_chat()
        st.rerun()
    if sidebar_action == "switch" and thread_id:
        _switch_thread(thread_id)
        st.rerun()

    active_thread = _active_thread()
    messages = active_thread["messages"]
    last_state = active_thread["last_state"]

    main_col, side_col = st.columns([0.74, 0.26], gap="large")

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
        pending_query: str | None = None
        if messages:
            render_messages(messages)
        else:
            pending_query = render_empty_state()

    with side_col:
        render_evidence_panel(last_state)

    if messages:
        chat_query = st.chat_input("输入公司、指标、财报或多步推理问题")
        if chat_query:
            pending_query = chat_query

    if pending_query:
        handle_query(pending_query)
        st.rerun()


if __name__ == "__main__":
    # 判断是否在Streamlit运行环境中
    _is_streamlit = False
    try:
        import streamlit.runtime.scriptrunner as _st_runner
        if _st_runner.get_script_run_ctx():
            _is_streamlit = True
    except Exception:
        pass

    # 检查sys.argv是否有streamlit特征（防止scriptrunner异常误判）
    import sys
    if not _is_streamlit and any("streamlit" in arg for arg in sys.argv):
        _is_streamlit = True

    if _is_streamlit:
        run_streamlit()
    else:
        run_cli()