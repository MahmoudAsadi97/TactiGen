"""
Lightweight Temporal Transformer for Football Action Anticipation.
Architecture inspired by FAANTRA (Transformer-based football action anticipation).
Uses ResNet18 frame features + Transformer Encoder + classification head.
"""
import math
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import List
from loguru import logger

from action_anticipation.anticipation_schema import AnticipationResult, ACTION_CLASSES
from tactical_analysis.tactical_metrics_schema import TacticalMetrics
from action_anticipation.baseline_models import HeuristicBaseline


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FootballActionTransformer(nn.Module):
    def __init__(self, feature_dim: int = 512, d_model: int = 256,
                 nhead: int = 4, num_layers: int = 2, num_classes: int = 9):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=512, dropout=0.1,
                                                    batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class FAAntraInference:
    def __init__(self, checkpoint_path: str = "outputs/models/football_transformer.pt",
                 sequence_len: int = 8, device: str = "cpu"):
        self.sequence_len = sequence_len
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        # Feature extractor: ResNet18 without FC layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor.eval().to(self.device)

        # Transformer model
        self.model = FootballActionTransformer().to(self.device)
        if self.checkpoint_path.exists():
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            logger.info(f"Loaded transformer checkpoint from {self.checkpoint_path}")
        else:
            logger.warning("No checkpoint found — using untrained transformer. Heuristic baseline will be primary.")

        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.heuristic = HeuristicBaseline()

    def extract_frame_features(self, frame_paths: List[str]) -> torch.Tensor:
        features = []
        for fp in frame_paths[:self.sequence_len]:
            img = Image.open(fp).convert("RGB")
            t = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                f = self.feature_extractor(t).squeeze()
            features.append(f)
        while len(features) < self.sequence_len:
            features.append(features[-1] if features else torch.zeros(512).to(self.device))
        return torch.stack(features).unsqueeze(0)   # [1, T, 512]

    def predict(self, clip_id: str, frame_paths: List[str], timestamp: float,
                metrics: TacticalMetrics = None) -> AnticipationResult:
        if not frame_paths or not self.checkpoint_path.exists():
            if metrics:
                return self.heuristic.predict(clip_id, timestamp, metrics)
            return AnticipationResult(
                clip_id=clip_id, timestamp=timestamp,
                predicted_action="pass", confidence=0.42,
                model_used="heuristic_fallback"
            )

        features = self.extract_frame_features(frame_paths)
        with torch.no_grad():
            logits = self.model(features)
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        pred_idx = int(torch.argmax(logits).item())
        confidence = float(max(probs))
        class_probs = {ACTION_CLASSES[i]: round(probs[i], 4) for i in range(len(ACTION_CLASSES))}

        return AnticipationResult(
            clip_id=clip_id,
            timestamp=timestamp,
            predicted_action=ACTION_CLASSES[pred_idx],
            confidence=round(confidence, 4),
            model_used="football_transformer",
            class_probabilities=class_probs
        )
