"""
TactiGen storage layer.

DatabaseManager connects to PostgreSQL via SQLAlchemy when it is reachable and
transparently falls back to local JSON-lines file storage when the database is
unavailable, so the pipeline never fails purely because Postgres is offline.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

try:
    from sqlalchemy import create_engine, text
    _SQLALCHEMY_AVAILABLE = True
except Exception:  # SQLAlchemy not installed
    _SQLALCHEMY_AVAILABLE = False

DEFAULT_POSTGRES_URL = "postgresql://tactigen:tactigen@localhost:5432/tactigen"
FALLBACK_DIR = Path("outputs/db_fallback")


class DatabaseManager:
    def __init__(self, postgres_url: str = None, fallback_dir: str = None):
        self.postgres_url = postgres_url or os.getenv("POSTGRES_URL", DEFAULT_POSTGRES_URL)
        self.fallback_dir = Path(fallback_dir) if fallback_dir else FALLBACK_DIR
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        self.engine = None
        self._connect()

    def _connect(self):
        if not _SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not installed — using JSON fallback storage.")
            return
        try:
            engine = create_engine(self.postgres_url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.engine = engine
            self._ensure_schema()
            logger.info("Connected to PostgreSQL.")
        except Exception as e:
            self.engine = None
            logger.warning(f"PostgreSQL unavailable ({e}). JSON fallback at {self.fallback_dir}.")

    def _ensure_schema(self):
        """Apply storage/schemas.sql (idempotent CREATE TABLE IF NOT EXISTS statements)."""
        schema_path = Path("storage/schemas.sql")
        if self.engine is None or not schema_path.exists():
            return
        sql = schema_path.read_text(encoding="utf-8")
        try:
            with self.engine.begin() as conn:
                for statement in [s.strip() for s in sql.split(";") if s.strip()]:
                    conn.execute(text(statement))
        except Exception as e:
            logger.warning(f"Schema application skipped: {e}")

    def _fallback_write(self, table: str, record: dict) -> str:
        path = self.fallback_dir / f"{table}.jsonl"
        payload = {**record, "_inserted_at": datetime.utcnow().isoformat()}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
        logger.info(f"Stored record in fallback file: {path}")
        return str(path)

    def insert_clip(self, clip_id: str, file_path: str = None, fps: float = None,
                    duration_seconds: float = None, resolution: str = None):
        record = {
            "clip_id": clip_id, "file_path": file_path, "fps": fps,
            "duration_seconds": duration_seconds, "resolution": resolution,
        }
        if self.engine is None:
            return self._fallback_write("raw_video_clips", record)
        try:
            with self.engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO raw_video_clips "
                    "(clip_id, file_path, fps, duration_seconds, resolution) "
                    "VALUES (:clip_id, :file_path, :fps, :duration_seconds, :resolution) "
                    "ON CONFLICT (clip_id) DO UPDATE SET "
                    "file_path = EXCLUDED.file_path, fps = EXCLUDED.fps, "
                    "duration_seconds = EXCLUDED.duration_seconds, resolution = EXCLUDED.resolution"
                ), record)
            return clip_id
        except Exception as e:
            logger.warning(f"insert_clip DB write failed ({e}); using fallback.")
            return self._fallback_write("raw_video_clips", record)

    def insert_summary(self, clip_id: str, main_phase: str = None, primary_pattern: str = None,
                       anticipated_action: str = None, confidence: float = None,
                       report_text: str = None, structured_report: dict = None,
                       report_status: str = "ok"):
        record = {
            "clip_id": clip_id, "main_phase": main_phase, "primary_pattern": primary_pattern,
            "anticipated_action": anticipated_action, "confidence": confidence,
            "report_text": report_text,
            "structured_report": json.dumps(structured_report) if structured_report is not None else None,
            "report_status": report_status,
        }
        if self.engine is None:
            return self._fallback_write("mart_clip_tactical_summary", record)
        try:
            with self.engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO mart_clip_tactical_summary "
                    "(clip_id, main_phase, primary_pattern, anticipated_action, confidence, "
                    " report_text, structured_report, report_status) "
                    "VALUES (:clip_id, :main_phase, :primary_pattern, :anticipated_action, :confidence, "
                    " :report_text, CAST(:structured_report AS JSONB), :report_status) "
                    "ON CONFLICT (clip_id) DO UPDATE SET "
                    "main_phase = EXCLUDED.main_phase, primary_pattern = EXCLUDED.primary_pattern, "
                    "anticipated_action = EXCLUDED.anticipated_action, confidence = EXCLUDED.confidence, "
                    "report_text = EXCLUDED.report_text, structured_report = EXCLUDED.structured_report, "
                    "report_status = EXCLUDED.report_status"
                ), record)
            return clip_id
        except Exception as e:
            logger.warning(f"insert_summary DB write failed ({e}); using fallback.")
            return self._fallback_write("mart_clip_tactical_summary", record)

    def insert_feedback(self, clip_id: str, report_id: str = None,
                        pattern_accuracy_score: int = None,
                        recommendation_usefulness_score: int = None,
                        hallucination_flag: bool = False, reviewer_comment: str = None):
        record = {
            "clip_id": clip_id, "report_id": report_id,
            "pattern_accuracy_score": pattern_accuracy_score,
            "recommendation_usefulness_score": recommendation_usefulness_score,
            "hallucination_flag": hallucination_flag, "reviewer_comment": reviewer_comment,
        }
        if self.engine is None:
            return self._fallback_write("analyst_feedback", record)
        try:
            with self.engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO analyst_feedback "
                    "(clip_id, report_id, pattern_accuracy_score, recommendation_usefulness_score, "
                    " hallucination_flag, reviewer_comment) "
                    "VALUES (:clip_id, :report_id, :pattern_accuracy_score, "
                    " :recommendation_usefulness_score, :hallucination_flag, :reviewer_comment)"
                ), record)
            return clip_id
        except Exception as e:
            logger.warning(f"insert_feedback DB write failed ({e}); using fallback.")
            return self._fallback_write("analyst_feedback", record)

    def close(self):
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
