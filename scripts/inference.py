"""Inference script for detection on test data.

Week 2-3: Run detector on scalograms, output JSON with alerts.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from ligo_gw.models import BaselineCNN
from ligo_gw.inference import GWDetector
from ligo_gw.analysis import ROCAnalyzer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Run inference on data."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    model = BaselineCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logger.info(f"Loaded model: {args.checkpoint}")
    
    # Create detector
    detector = GWDetector(
        model,
        device=device,
        threshold=args.threshold
    )
    
    # TODO: Load test data
    # X_test = load_scalograms(args.data_dir)
    
    # For now: dummy data
    X_test = np.random.randn(100, 1, 128, 256).astype(np.float32)
    logger.info(f"Running inference on {X_test.shape[0]} scalograms...")
    
    # Inference
    detections, confidences = detector.predict(X_test)
    
    # Prepare output
    results = {
        "num_detections": int(np.sum(detections)),
        "alerts": []
    }
    
    for idx, (detection, confidence) in enumerate(zip(detections, confidences)):
        if detection:
            results["alerts"].append({
                "event_id": idx,
                "confidence": float(confidence),
                "gps_time": None  # TODO: Add GPS time
            })
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results: {output_path}")
    logger.info(f"Found {results['num_detections']} detections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GW detection inference")
    
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--output-file", default="./detections.json", help="Output JSON file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    
    args = parser.parse_args()
    main(args)
