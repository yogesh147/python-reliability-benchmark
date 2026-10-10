import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Kaggle CodeBench AI",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.6rem;
    max-width: 1500px;
}
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.18);
}
.hero {
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(120deg, #14243b, #24557a);
    color: white;
    margin-bottom: 22px;
}
.hero p { color: #d6e5f5; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Kaggle CodeBench AI")
    st.caption("Python Model Benchmark Lab")
    page = st.radio(
        "Navigation",
        ["Overview", "Leaderboard", "Task Explorer", "Run Benchmark"],
    )
    st.divider()
    st.caption("Results are based on recorded model runs.")


def load_results():
    import pandas as pd
    from pathlib import Path

    results_dir = Path(__file__).parent / "results"
    csv_path = results_dir / "raw_results.csv"

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame(
            columns=[
                "model",
                "task_id",
                "category",
                "passed_tests",
                "total_tests",
                "pass_rate",
                "syntax_valid",
                "constraint_compliant",
                "error",
                "latency_seconds",
            ]
        )

    try:
        return pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(
            columns=[
                "model",
                "task_id",
                "category",
                "passed_tests",
                "total_tests",
                "pass_rate",
                "syntax_valid",
                "constraint_compliant",
                "error",
                "latency_seconds",
            ]
        )


def load_tasks():
    with open(ROOT / "tasks.json", encoding="utf-8") as f:
        return json.load(f)
    return None

df = load_results()

st.markdown("""
<div class="hero">
  <h1>Python Benchmark Lab</h1>
  <p>Measure how reliably AI models write, debug, and explain code.
  Compare test pass rates, latency, and instruction compliance.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    if df.empty:
        st.info(
            "No evaluation results yet. Configure your model APIs "
            "and run the benchmark to populate this dashboard."
        )

    a, b, c, d = st.columns(4)

    a.metric(
        "Models evaluated",
        df["model"].nunique() if not df.empty else 0,
    )
    b.metric(
        "Tasks evaluated",
        len(df) if not df.empty else 0,
    )
    c.metric(
        "Mean test pass rate",
        f"{df['pass_rate'].mean():.1%}"
        if not df.empty else "—",
    )
    d.metric(
        "Mean latency",
        f"{df['latency_seconds'].mean():.2f}s"
        if not df.empty else "—",
    )

    st.subheader("Performance by category")

    if not df.empty:
        category = (
            df.groupby(["model", "category"], as_index=False)
            ["pass_rate"].mean()
        )

        fig = px.bar(
            category,
            x="category",
            y="pass_rate",
            color="model",
            barmode="group",
            range_y=[0, 1],
            labels={
                "category": "Benchmark category",
                "pass_rate": "Mean test pass rate",
                "model": "AI model",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Latest evaluation records")
        st.dataframe(
            df.sort_values("latency_seconds"),
            use_container_width=True,
            hide_index=True,
        )

elif page == "Leaderboard":
    st.subheader("🏆 Model leaderboard")

    if df.empty:
        st.warning("Run an evaluation to generate rankings.")
    else:
        board = (
            df.groupby("model", as_index=False)
            .agg(
                Tasks=("task_id", "count"),
                Pass_Rate=("pass_rate", "mean"),
                Syntax_Validity=("syntax_valid", "mean"),
                Avg_Latency=("latency_seconds", "mean"),
            )
            .sort_values("Pass_Rate", ascending=False)
        )

        board["Pass_Rate"] = (board["Pass_Rate"] * 100).round(2)
        board["Syntax_Validity"] = (
                board["Syntax_Validity"] * 100
        ).round(2)
        board["Avg_Latency"] = board["Avg_Latency"].round(3)

        board.index = range(1, len(board) + 1)
        board.index.name = "Rank"

        st.dataframe(board, use_container_width=True)

        st.download_button(
            "Download leaderboard CSV",
            board.to_csv().encode("utf-8"),
            file_name="leaderboard.csv",
            mime="text/csv",
        )

elif page == "Task Explorer":
    st.subheader("🔎 Explore benchmark tasks")

    tasks = load_tasks()
    task_df = pd.DataFrame(tasks)

    category = st.selectbox(
        "Filter by category",
        ["All"] + sorted(task_df["category"].unique().tolist()),
        )

    if category != "All":
        task_df = task_df[task_df["category"] == category]

    for _, task in task_df.iterrows():
        with st.expander(
                f"{task['id']} · {task['category']}"
        ):
            st.write(task["prompt"])
            st.caption(
                f"Function: {task['function_name']} · "
                f"{len(task['tests'])} test cases"
            )

            st.json(task["tests"])

elif page == "Run Benchmark":
    st.subheader("▶ Run an evaluation")
    st.write(
        "This invokes benchmark.py using the model configuration "
        "already set in your environment."
    )

    st.warning(
        "API calls may incur charges. Check your model configuration "
        "and execution environment before starting."
    )

    config = os.environ.get("MODELS", "[]")

    try:
        models = json.loads(config)
        st.write(
            "Configured models:",
            [m.get("name", "Unnamed") for m in models],
        )
    except json.JSONDecodeError:
        st.error("MODELS is not valid JSON.")
        models = []

    if st.button(
            "Start benchmark",
            type="primary",
            disabled=not models,
    ):
        with st.status(
                "Evaluating models...",
                expanded=True,
        ) as status:
            try:
                process = subprocess.Popen(
                    [sys.executable, str(ROOT / "benchmark.py")],
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                log = st.empty()
                lines = []

                for line in process.stdout:
                    lines.append(line.rstrip())
                    log.code("\n".join(lines[-30:]))

                return_code = process.wait()

                if return_code == 0:
                    status.update(
                        label="Benchmark completed",
                        state="complete",
                    )
                    st.success(
                        "Evaluation finished. Open Overview "
                        "or Leaderboard to inspect the results."
                    )
                    st.rerun()
                else:
                    status.update(
                        label="Benchmark failed",
                        state="error",
                    )
                    st.error("Check the logs and API configuration.")

            except Exception as exc:
                status.update(
                    label="Could not start benchmark",
                    state="error",
                )
                st.exception(exc)

st.divider()
st.caption(
    "CodeBench AI · Python Reliability Benchmark · "
    "Report measured results only."
)