"""TactiGen full analysis pipeline."""
import json
import time
from datetime import datetime
from pathlib import Path
from loguru import logger
from orchestration.agents import (
    PreprocessingAgent, LocalizationAgent, ActionPredictionAgent,
    TacticalAnalysisAgent, EvidenceCompilerAgent, RAGRetrievalAgent,
    ReportGenerationAgent, VisualizationFeedbackAgent
)


class TactiGenPipeline:
    def __init__(self):
        self.agents = [
            ("Preprocessing", PreprocessingAgent()),
            ("Localization", LocalizationAgent()),
            ("ActionPrediction", ActionPredictionAgent()),
            ("TacticalAnalysis", TacticalAnalysisAgent()),
            ("EvidenceCompiler", EvidenceCompilerAgent()),
            ("RAGRetrieval", RAGRetrievalAgent()),
            ("ReportGeneration", ReportGenerationAgent()),
            ("VisualizationFeedback", VisualizationFeedbackAgent()),
        ]

    def run(self, video_path: str = None, clip_id: str = None) -> dict:
        if clip_id is None:
            clip_id = f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        context = {
            "clip_id": clip_id,
            "video_path": video_path,
            "limitations": [],
            "run_started_at": datetime.utcnow().isoformat()
        }
        logger.info(f"=== TactiGen Pipeline starting: {clip_id} ===")
        timings = {}

        for name, agent in self.agents:
            t0 = time.time()
            try:
                context = agent.run(context)
            except Exception as e:
                logger.error(f"Agent {name} failed: {e}")
                context["limitations"].append(f"{name} agent failed: {str(e)}")
            timings[name] = round(time.time() - t0, 3)
            logger.info(f"  ok {name} ({timings[name]}s)")

        # Save run log
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clip_id}"
        run_log = {
            "run_id": run_id,
            "clip_id": clip_id,
            "video_file": video_path or "synthetic",
            "report_status": context.get("evidence", {}).get("report_status", "unknown"),
            "agent_timings_seconds": timings,
            "generated_at": datetime.utcnow().isoformat()
        }
        Path("outputs/run_logs").mkdir(parents=True, exist_ok=True)
        with open(f"outputs/run_logs/{run_id}.json", "w") as f:
            json.dump(run_log, f, indent=2)

        # MLflow experiment tracking (non-critical: degrades gracefully if unavailable)
        try:
            import mlflow
            with mlflow.start_run(run_name=run_id):
                mlflow.log_param("clip_id", clip_id)
                mlflow.log_param("localization_model", "yolov8m")
                mlflow.log_param("anticipation_model", "football_transformer")
                mlflow.log_metric("localization_confidence", context.get("localization_confidence", 0))
                mlflow.log_metric("anticipation_confidence",
                                  context.get("anticipation_result", {}).get("confidence", 0))
                mlflow.log_metric("overload_ratio",
                                  context.get("tactical_metrics", {}).get("overload_ratio", 0))
                mlflow.log_metric("compactness_score",
                                  context.get("tactical_metrics", {}).get("compactness_score", 0))
                report_path = f"outputs/reports/{clip_id}_report.txt"
                if Path(report_path).exists():
                    mlflow.log_artifact(report_path)
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-critical): {e}")

        logger.success(f"=== Pipeline complete: {clip_id} ===")
        return context
