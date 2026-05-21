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