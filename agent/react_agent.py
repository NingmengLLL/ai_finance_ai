from __future__ import annotations

from agent.graph import FinancialGraphAgent


class ReactAgent(FinancialGraphAgent):
    """Backward-compatible alias for the old Streamlit entrypoint."""

    pass


if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream("示例科技2024年现金流质量如何？"):
        print(chunk, end="", flush=True)
