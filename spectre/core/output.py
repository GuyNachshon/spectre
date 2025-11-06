"""Output generation for fingerprints, CSV, and visualizations."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from spectre.core.features import FeatureStore
from spectre.core.pipeline import Pipeline
from spectre.core.scoring import Scorer
from spectre.io.loader import load_checkpoint
from spectre.viz.graph import plot_graph_signal_map, plot_sequence_changepoints, plot_tda_barcode
from spectre.viz.heatmaps import plot_similarity_heatmap, plot_zscore_heatmap
from spectre.viz.spectra import plot_mp_fit, plot_spectral_spectrum
from spectre.viz.spectrogram import plot_weight_spectrogram


def generate_fingerprint(
    pipeline: Pipeline,
    output_dir: Path
) -> Path:
    """
    Generate fingerprint_v2.json.
    
    Args:
        pipeline: Pipeline instance
        output_dir: Output directory
        
    Returns:
        Path to fingerprint file
    """
    results = pipeline._generate_results()
    
    fingerprint_path = output_dir / "fingerprint_v2.json"
    
    with open(fingerprint_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return fingerprint_path


def generate_features_csv(
    feature_store: FeatureStore,
    output_dir: Path
) -> Path:
    """
    Generate features.csv.
    
    Args:
        feature_store: FeatureStore instance
        output_dir: Output directory
        
    Returns:
        Path to CSV file
    """
    csv_path = output_dir / "features.csv"
    
    # Get all feature names
    all_feature_names = feature_store.get_all_feature_names()
    
    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        header = ["name", "role", "layer_idx", "shape"] + all_feature_names
        writer.writerow(header)
        
        # Rows
        for tf in feature_store.features:
            row = [
                tf.name,
                tf.role.value,
                tf.layer_idx if tf.layer_idx is not None else "",
                str(tf.shape),
            ]
            row.extend([tf.features.get(fn, "") for fn in all_feature_names])
            writer.writerow(row)
    
    return csv_path


def generate_per_layer_summary(
    feature_store: FeatureStore,
    scorer: Optional[Scorer],
    output_dir: Path
) -> Path:
    """
    Generate per_layer_summary.csv.
    
    Args:
        feature_store: FeatureStore instance
        scorer: Optional Scorer instance
        output_dir: Output directory
        
    Returns:
        Path to CSV file
    """
    csv_path = output_dir / "per_layer_summary.csv"
    
    # Group by layer
    layers = {}
    for tf in feature_store.features:
        if tf.layer_idx is not None:
            layer_idx = tf.layer_idx
            if layer_idx not in layers:
                layers[layer_idx] = []
            layers[layer_idx].append(tf)
    
    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        header = [
            "layer_idx",
            "num_tensors",
            "avg_ensemble_score",
            "max_ensemble_score",
            "flag",
        ]
        writer.writerow(header)
        
        # Rows
        for layer_idx in sorted(layers.keys()):
            layer_tensors = layers[layer_idx]
            
            # Compute statistics
            num_tensors = len(layer_tensors)
            
            if scorer:
                ensemble_scores = [
                    scorer.ensemble_scores.get(tf.name, 0.0)
                    for tf in layer_tensors
                ]
                avg_score = np.mean(ensemble_scores) if ensemble_scores else 0.0
                max_score = np.max(ensemble_scores) if ensemble_scores else 0.0
                
                # Get worst flag
                flags = [scorer.get_flag(tf.name) for tf in layer_tensors]
                if "HARD_RED" in flags:
                    flag = "HARD_RED"
                elif "RED" in flags:
                    flag = "RED"
                elif "AMBER" in flags:
                    flag = "AMBER"
                else:
                    flag = "GREEN"
            else:
                avg_score = 0.0
                max_score = 0.0
                flag = "GREEN"
            
            writer.writerow([
                layer_idx,
                num_tensors,
                avg_score,
                max_score,
                flag,
            ])
    
    return csv_path


def generate_visualizations(
    pipeline: Pipeline,
    output_dir: Path,
    top_k: int = 20
):
    """
    Generate visual artifacts.
    
    Args:
        pipeline: Pipeline instance
        output_dir: Output directory
        top_k: Number of top suspects for visualizations
    """
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Heatmaps
    if pipeline.scorer:
        plot_similarity_heatmap(pipeline.scorer, viz_dir / "similarity_heatmap.png", top_k=top_k)
        plot_zscore_heatmap(pipeline.scorer, viz_dir / "zscore_heatmap.png", top_k=top_k)
    
    # Get top suspects for detailed visualizations
    if pipeline.scorer:
        suspects = pipeline.scorer.get_suspects(top_k=top_k)
    else:
        suspects = []
    
    # Generate per-tensor visualizations for top suspects
    for suspect in suspects[:top_k]:
        tensor_name = suspect["name"]
        tf = pipeline.feature_store.get(tensor_name)
        
        if tf is None:
            continue
        
        # Load tensor data
        checkpoint_path = Path(pipeline.config.get("model.path"))
        if checkpoint_path is None:
            continue
        
        # Find tensor in checkpoint
        for name, array in load_checkpoint(checkpoint_path):
            if name == tensor_name:
                # Spectral spectrum
                layer_suffix = f"_{tf.layer_idx}" if tf.layer_idx is not None else ""
                plot_spectral_spectrum(
                    tf,
                    array,
                    viz_dir / f"spectral_spectrum_{tensor_name[:30]}{layer_suffix}.png"
                )
                
                # MP fit
                plot_mp_fit(
                    tf,
                    array,
                    viz_dir / f"mp_fit_{tensor_name[:30]}{layer_suffix}.png"
                )
                
                # Spectrogram
                plot_weight_spectrogram(
                    tf,
                    array,
                    viz_dir / f"weight_spectrogram_{tensor_name[:30]}{layer_suffix}.png"
                )
                
                break
    
    # Role-based visualizations
    from spectre.io.name_mapper import ParameterRole
    
    for role in ParameterRole:
        if pipeline.scorer:
            plot_graph_signal_map(role, pipeline.feature_store, viz_dir / f"graph_signal_map_{role.value}.png")
            plot_sequence_changepoints(role, pipeline.feature_store, viz_dir / f"sequence_changepoints_{role.value}.png")


def generate_all_outputs(
    pipeline: Pipeline,
    output_dir: Path,
    save_visuals: bool = True,
    topk_visuals: int = 20
):
    """
    Generate all outputs: fingerprint, CSV, visualizations.
    
    Args:
        pipeline: Pipeline instance
        output_dir: Output directory
        save_visuals: Whether to save visualizations
        topk_visuals: Number of top suspects for visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate fingerprint
    generate_fingerprint(pipeline, output_dir)
    
    # Generate CSV files
    generate_features_csv(pipeline.feature_store, output_dir)
    generate_per_layer_summary(pipeline.feature_store, pipeline.scorer, output_dir)
    
    # Generate visualizations
    if save_visuals:
        generate_visualizations(pipeline, output_dir, top_k=topk_visuals)

