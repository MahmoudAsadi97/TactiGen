"""TactiGen Agent Orchestration Layer."""
import json
from pathlib import Path
from loguru import logger
from typing import List
import numpy as np


class PreprocessingAgent:
    def run(self, context: dict) -> dict:
        logger.info("[Agent] Preprocessing...")
        from preprocessing.extract_frames import FrameExtractor
        from preprocessing.segment_clip import ClipSegmenter

        video_path = context.get("video_path")
        clip_id = context.get("clip_id", "clip_000")

        if video_path and Path(video_path).exists():
            extractor = FrameExtractor(sample_rate=5, output_dir=f"outputs/frames/{clip_id}")
            meta = extractor.extract(video_path)
            segmenter = ClipSegmenter(fps=meta["fps"])
            segments = segmenter.segment(meta["duration_seconds"], clip_id_prefix=clip_id)
            context["frame_paths"] = meta["frame_paths"]
            context["video_meta"] = meta
            context["segments"] = segments
        else:
            # Synthetic fallback
            synthetic_path = Path("data/sample_clips/synthetic_clip_001.json")
            context["frame_paths"] = []
            context["video_meta"] = {"fps": 5, "duration_seconds": 5.0, "total_frames": 25}
            context["segments"] = [{"clip_id": clip_id, "start_time": 0.0, "end_time": 5.0,
                                    "frame_start": 0, "frame_end": 24}]
            context["synthetic_path"] = str(synthetic_path)
        return context


class LocalizationAgent:
    def run(self, context: dict) -> dict:
        logger.info("[Agent] Localizing players...")
        from localization.player_localization import PlayerLocalizer
        localizer = PlayerLocalizer()
        synthetic_path = context.get("synthetic_path")

        if synthetic_path and Path(synthetic_path).exists():
            records = localizer.localize_from_synthetic(synthetic_path)
        else:
            records = []
            fps = context["video_meta"].get("fps", 25)
            for i, fp in enumerate(context.get("frame_paths", [])):
                ts = i * (5 / fps) if fps else 0
                records.extend(localizer.localize_frame(fp, i * 5, ts, context["clip_id"]))

        context["localization_records"] = [r.model_dump() for r in records]
        context["localization_confidence"] = (
            float(np.mean([r.get("confidence", 0.7) for r in context["localization_records"]]))
            if context["localization_records"] else 0.7
        )
        return context


class ActionPredictionAgent:
    def run(self, context: dict) -> dict:
        logger.info("[Agent] Predicting action...")
        timestamp = context.get("segments", [{}])[0].get("start_time", 0.0)
        try:
            from action_anticipation.faantra_inference import FAAntraInference
            predictor = FAAntraInference()
            result = predictor.predict(
                clip_id=context["clip_id"],
                frame_paths=context.get("frame_paths", []),
                timestamp=timestamp,
            )
            context["anticipation_result"] = result.model_dump()
        except Exception as e:
            logger.warning(f"Transformer path unavailable ({e}); using heuristic anticipation fallback.")
            from action_anticipation.anticipation_schema import AnticipationResult
            context["anticipation_result"] = AnticipationResult(
                clip_id=context["clip_id"], timestamp=timestamp,
                predicted_action="pass", confidence=0.42,
                model_used="heuristic_fallback",
            ).model_dump()
        return context


class TacticalAnalysisAgent:
    def run(self, context: dict) -> dict:
        logger.info("[Agent] Computing tactical metrics...")
        from tactical_analysis.team_width import compute_team_width
        from tactical_analysis.compactness import compute_compactness
        from tactical_analysis.defensive_line import compute_defensive_line_height
        from tactical_analysis.overload_detection import detect_worst_overload
        from tactical_analysis.tactical_metrics_schema import TacticalMetrics

        records = context.get("localization_records", [])
        positions_all = [(r["x_world"], r["y_world"]) for r in records if r.get("x_world") is not None]

        # Split teams heuristically: left half = team B (defending), right half = team A (attacking)
        mid_x = 52.5
        team_a = [p for p in positions_all if p[0] >= mid_x]
        team_b = [p for p in positions_all if p[0] < mid_x]

        segment = context.get("segments", [{"start_time": 0.0, "end_time": 5.0}])[0]
        overload_info = detect_worst_overload(team_a, team_b)

        metrics = TacticalMetrics(
            clip_id=context["clip_id"],
            time_window_start=segment["start_time"],
            time_window_end=segment["end_time"],
            team_width_meters=compute_team_width(team_a),
            compactness_score=compute_compactness(team_b),
            defensive_line_height_meters=compute_defensive_line_height(team_b),
            overload_channel=overload_info["channel"],
            attacking_count=overload_info["attacking_count"],
            defending_count=overload_info["defending_count"],
            overload_ratio=overload_info["overload_ratio"]
        )
        context["tactical_metrics"] = metrics.model_dump()
        return context


class EvidenceCompilerAgent:
    def run(self, context: dict) -> dict:
        logger.info("[Agent] Compiling evidence...")
        from evidence.evidence_compiler import EvidenceCompiler
        from tactical_analysis.tactical_metrics_schema import TacticalMetrics
        from action_anticipation.anticipation_schema import AnticipationResult

        compiler = EvidenceCompiler()
        metrics = TacticalMetrics(**context["tactical_metrics"])
        anticipation = AnticipationResult(**context["anticipation_result"])
        segment = context.get("segments", [{"start_time": 0.0, "end_time": 5.0}])[0]

        evidence = compiler.compile(
            clip_id=context["clip_id"],
            time_window=(segment["start_time"], segment["end_time"]),
            metrics=metrics,
            localization_confidence=context.get("localization_confidence", 0.75),
            anticipation_result=anticipation,
            supporting_frame_paths=context.get("frame_paths", [])[:3],
            limitations=context.get("limitations", [])
        )
        context["evidence"] = evidence.model_dump()
        return context


class RAGRetrievalAgent:
    def __init__(self):
        self._retriever = None

    def run(self, context: dict) -> dict:
        logger.info("[Agent] Retrieving tactical knowledge...")
        if self._retriever is None:
            from rag.retrieve_context import TacticalRetriever
            self._retriever = TacticalRetriever()
        pattern = context["evidence"].get("detected_pattern", "general_possession")
        passages = self._retriever.retrieve_for_pattern(pattern)
        context["retrieved_passages"] = passages
        return context


class ReportGenerationAgent:
    def run(self, context: dict) -> dict:
        logger.info("[Agent] Generating reports...")
        import yaml
        from evidence.evidence_schema import EvidenceObject
        from report_generation.generate_structured_report import generate_structured_report
        from report_generation.generate_text_report import generate_text_report

        ontology = {}
        onto_path = Path("rag/tactical_ontology.yaml")
        if onto_path.exists():
            with open(onto_path) as f:
                ontology = yaml.safe_load(f)

        evidence = EvidenceObject(**context["evidence"])
        structured = generate_structured_report(evidence)
        text_report = generate_text_report(
            evidence, context.get("retrieved_passages", []), structured, ontology
        )
        context["structured_report"] = structured
        context["text_report"] = text_report
        return context


class VisualizationFeedbackAgent:
    def run(self, context: dict) -> dict:
        logger.info("[Agent] Generating visualizations...")
        from visualization.heatmaps import generate_tactical_heatmap
        from visualization.trajectories import generate_trajectory_plot

        records = context.get("localization_records", [])
        positions = [(r["x_world"], r["y_world"]) for r in records if r.get("x_world") is not None]

        heatmap_path = generate_tactical_heatmap(
            positions, title=f"Tactical Heatmap — {context['clip_id']}"
        )
        # Build player tracks
        tracks = {}
        for r in records:
            pid = r.get("player_id", 0)
            if r.get("x_world") is not None:
                tracks.setdefault(pid, []).append((r["x_world"], r["y_world"]))

        traj_path = generate_trajectory_plot(
            tracks, title=f"Trajectories — {context['clip_id']}"
        )
        context["heatmap_path"] = heatmap_path
        context["trajectory_path"] = traj_path
        return context
