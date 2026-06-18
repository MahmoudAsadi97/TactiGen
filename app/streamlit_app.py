"""TactiGen Streamlit Application."""
import sys
import os

# Ensure the repository root is importable when launched via `streamlit run app/streamlit_app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="TactiGen — Football Tactical AI", layout="wide", page_icon="⚽")

# Sidebar
st.sidebar.title("⚽ TactiGen")
st.sidebar.markdown("Evidence-Grounded Football Tactical AI")
uploaded_file = st.sidebar.file_uploader("Upload football clip", type=["mp4", "mov", "avi"])
anticipation_window = st.sidebar.slider("Anticipation window (seconds)", 3, 10, 5)
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.50, 0.90, 0.65, step=0.05)
run_button = st.sidebar.button("▶ Run Analysis", type="primary")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📹 Video & Frames",
    "👥 Localization",
    "📐 Tactical Metrics",
    "🎯 Action Anticipation",
    "📋 Tactical Report",
    "✍️ Analyst Feedback"
])

context = None

if run_button:
    from orchestration.pipeline import TactiGenPipeline

    clip_id = "streamlit_clip"
    video_path = None

    if uploaded_file:
        Path("outputs/uploads").mkdir(parents=True, exist_ok=True)
        video_path = f"outputs/uploads/{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

    with st.status("Running TactiGen pipeline...", expanded=True) as status:
        st.write("Preprocessing video...")
        pipeline = TactiGenPipeline()
        st.write("Running all agents...")
        context = pipeline.run(video_path=video_path, clip_id=clip_id)
        status.update(label="Analysis complete!", state="complete")
    st.session_state["context"] = context

if "context" in st.session_state:
    context = st.session_state["context"]

    with tab1:
        st.subheader("Video & Sampled Frames")
        if context.get("video_path") and Path(context["video_path"]).exists():
            st.video(context["video_path"])
        frame_paths = context.get("frame_paths", [])[:4]
        if frame_paths:
            cols = st.columns(2)
            for i, fp in enumerate(frame_paths):
                if Path(fp).exists():
                    cols[i % 2].image(fp, caption=f"Frame {i+1}", use_column_width=True)
        else:
            st.info("No video frames available — using synthetic player data.")

    with tab2:
        st.subheader("Player Localization")
        records = context.get("localization_records", [])
        if records:
            import pandas as pd
            df = pd.DataFrame(records)[["frame_id", "timestamp", "player_id", "x_world", "y_world", "confidence"]]
            st.dataframe(df.head(20), use_container_width=True)
        avg_conf = context.get("localization_confidence", 0)
        st.metric("Average Localization Confidence", f"{avg_conf:.3f}",
                  delta="✓ Above threshold" if avg_conf >= confidence_threshold else "⚠ Below threshold")
        traj_path = context.get("trajectory_path")
        if traj_path and Path(traj_path).exists():
            st.image(traj_path, caption="Player Trajectories", use_column_width=True)

    with tab3:
        st.subheader("Tactical Metrics")
        metrics = context.get("tactical_metrics", {})
        if metrics:
            c1, c2 = st.columns(2)
            c1.metric("Team Width", f"{metrics.get('team_width_meters', 0):.1f} m")
            c2.metric("Compactness", f"{metrics.get('compactness_score', 0):.3f}")
            c1.metric("Defensive Line Height", f"{metrics.get('defensive_line_height_meters', 0):.1f} m")
            c2.metric("Overload Ratio", f"{metrics.get('overload_ratio', 0):.2f}",
                      delta=f"{metrics.get('overload_channel','?')} channel")
        heatmap_path = context.get("heatmap_path")
        if heatmap_path and Path(heatmap_path).exists():
            st.image(heatmap_path, caption="Tactical Heatmap", use_column_width=True)

    with tab4:
        st.subheader("Action Anticipation")
        ant = context.get("anticipation_result", {})
        if ant:
            st.metric("Predicted Action", ant.get("predicted_action", "N/A").upper())
            conf = ant.get("confidence", 0)
            st.progress(conf, text=f"Confidence: {conf:.2%}")
            st.caption(f"Model: {ant.get('model_used', 'N/A')} · "
                       f"Window: {ant.get('anticipation_window_seconds', 5)}s")
            if ant.get("class_probabilities"):
                import pandas as pd
                probs_df = pd.DataFrame(
                    list(ant["class_probabilities"].items()),
                    columns=["Action", "Probability"]
                ).sort_values("Probability", ascending=False)
                st.bar_chart(probs_df.set_index("Action"))

    with tab5:
        st.subheader("Tactical Report")
        evidence = context.get("evidence", {})
        if evidence:
            report_status = evidence.get("report_status", "ok")
            if report_status == "low_confidence_review":
                st.error("⚠️ LOW CONFIDENCE: This clip requires manual review before conclusions are drawn.")
            else:
                st.success(f"✓ Pattern detected: **{evidence.get('detected_pattern','').replace('_',' ').title()}**")

            with st.expander("Evidence Object (JSON)"):
                st.json(evidence)
        text_report = context.get("text_report", "No report generated.")
        st.text_area("Coaching Report", text_report, height=400)
        with st.expander("Structured JSON Report"):
            st.json(context.get("structured_report", {}))

    with tab6:
        st.subheader("Analyst Feedback")
        with st.form("feedback_form"):
            pattern_acc = st.select_slider("Pattern accuracy", options=[1, 2, 3, 4, 5], value=3)
            rec_useful = st.select_slider("Recommendation usefulness", options=[1, 2, 3, 4, 5], value=3)
            hallucination = st.checkbox("Did the system make unsupported claims?")
            comment = st.text_area("Reviewer comment")
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                import json
                feedback = {
                    "clip_id": context.get("clip_id"),
                    "pattern_accuracy_score": pattern_acc,
                    "recommendation_usefulness_score": rec_useful,
                    "hallucination_flag": hallucination,
                    "reviewer_comment": comment
                }
                Path("outputs/feedback").mkdir(parents=True, exist_ok=True)
                with open(f"outputs/feedback/{context.get('clip_id')}_feedback.json", "w") as f:
                    json.dump(feedback, f, indent=2)
                st.success("Feedback saved!")

else:
    for tab in [tab1, tab2, tab3, tab4, tab5, tab6]:
        with tab:
            st.info("Upload a clip and click **▶ Run Analysis** to begin.")
