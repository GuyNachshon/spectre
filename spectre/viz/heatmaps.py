"""Heatmap visualizations for similarity and z-scores."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from spectre.core.scoring import Scorer


def plot_similarity_heatmap(
    scorer: Scorer,
    output_path: Path,
    top_k: int = 50
):
    """
    Plot similarity heatmap between tensors.
    
    Args:
        scorer: Scorer instance
        output_path: Path to save figure
        top_k: Number of top tensors to include
    """
    # Get top-k suspects
    suspects = scorer.get_suspects(top_k=top_k)
    
    if len(suspects) < 2:
        return
    
    # Extract detector scores
    detector_names = list(scorer.weights.keys())
    tensor_names = [s["name"] for s in suspects]
    
    # Build similarity matrix
    similarity_matrix = np.zeros((len(tensor_names), len(tensor_names)))
    
    for i, tensor1 in enumerate(tensor_names):
        scores1 = scorer.detector_scores.get(tensor1, {})
        for j, tensor2 in enumerate(tensor_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                scores2 = scorer.detector_scores.get(tensor2, {})
                
                # Cosine similarity of detector scores
                vec1 = np.array([scores1.get(d, 0.0) for d in detector_names])
                vec2 = np.array([scores2.get(d, 0.0) for d in detector_names])
                
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                    similarity_matrix[i, j] = similarity
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        xticklabels=[name[:30] for name in tensor_names],
        yticklabels=[name[:30] for name in tensor_names],
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Similarity"}
    )
    plt.title("Tensor Similarity Heatmap")
    plt.xlabel("Tensor")
    plt.ylabel("Tensor")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_zscore_heatmap(
    scorer: Scorer,
    output_path: Path,
    top_k: int = 50
):
    """
    Plot z-score heatmap for detectors.
    
    Args:
        scorer: Scorer instance
        output_path: Path to save figure
        top_k: Number of top tensors to include
    """
    # Get top-k suspects
    suspects = scorer.get_suspects(top_k=top_k)
    
    if len(suspects) < 1:
        return
    
    # Extract detector scores
    detector_names = list(scorer.weights.keys())
    tensor_names = [s["name"] for s in suspects]
    
    # Build z-score matrix
    zscore_matrix = np.zeros((len(tensor_names), len(detector_names)))
    
    for i, tensor_name in enumerate(tensor_names):
        scores = scorer.detector_scores.get(tensor_name, {})
        for j, detector in enumerate(detector_names):
            zscore_matrix[i, j] = scores.get(detector, 0.0)
    
    # Plot heatmap
    plt.figure(figsize=(14, max(8, len(tensor_names) * 0.3)))
    sns.heatmap(
        zscore_matrix,
        xticklabels=detector_names,
        yticklabels=[name[:40] for name in tensor_names],
        cmap="RdYlGn_r",
        center=0,
        vmin=-5,
        vmax=5,
        linewidths=0.5,
        cbar_kws={"label": "Z-Score"}
    )
    plt.title("Detector Z-Score Heatmap")
    plt.xlabel("Detector")
    plt.ylabel("Tensor")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

