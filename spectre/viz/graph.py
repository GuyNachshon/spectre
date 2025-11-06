"""Graph visualizations."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

from spectre.core.features import TensorFeatures
from spectre.io.name_mapper import ParameterRole


def plot_graph_signal_map(
    role: ParameterRole,
    feature_store,
    output_path: Path
):
    """
    Plot graph signal map for a role.
    
    Args:
        role: ParameterRole
        feature_store: FeatureStore instance
        output_path: Path to save figure
    """
    role_features = feature_store.get_by_role(role)
    
    if len(role_features) < 2:
        return
    
    try:
        # Build graph from layer indices
        layers = [tf.layer_idx for tf in role_features if tf.layer_idx is not None]
        
        if len(layers) < 2:
            return
        
        # Create graph: edges between consecutive layers
        G = nx.Graph()
        for i, layer in enumerate(layers):
            G.add_node(i, layer=layer)
            if i > 0:
                G.add_edge(i - 1, i)
        
        # Get ensemble scores (if available)
        scores = []
        for tf in role_features:
            if tf.layer_idx is not None:
                # Use a feature as score proxy
                score = tf.features.get("spectral.stable_rank", 0.0)
                scores.append(score)
        
        # Plot graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Color nodes by score
        if scores:
            node_colors = scores
            nx.draw(
                G,
                pos,
                node_color=node_colors,
                node_size=500,
                cmap=plt.cm.RdYlGn_r,
                with_labels=True,
                labels={i: layers[i] for i in range(len(layers))},
                font_size=8
            )
        else:
            nx.draw(
                G,
                pos,
                node_size=500,
                with_labels=True,
                labels={i: layers[i] for i in range(len(layers))},
                font_size=8
            )
        
        plt.title(f"Graph Signal Map: {role.value}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def plot_tda_barcode(
    tensor_features: TensorFeatures,
    array: np.ndarray,
    output_path: Path
):
    """
    Plot TDA barcode (persistence diagram).
    
    Args:
        tensor_features: TensorFeatures instance
        array: Weight tensor
        output_path: Path to save figure
    """
    if not GUDHI_AVAILABLE:
        return
    
    if array.ndim < 2:
        return
    
    # This is a placeholder - full TDA barcode plotting would require
    # computing persistence and plotting birth/death pairs
    # For now, we'll create a simple visualization
    
    try:
        # Sample patches for TDA
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        
        m, n = array.shape
        
        if min(m, n) < 10:
            return
        
        # This is a simplified version
        # Full implementation would compute persistence and plot barcode
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"TDA Barcode for {tensor_features.name[:50]}\n(Full implementation requires persistence computation)", 
                ha="center", va="center", fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def plot_sequence_changepoints(
    role: ParameterRole,
    feature_store,
    output_path: Path,
    feature_name: str = "spectral.stable_rank"
):
    """
    Plot sequence change-points for a role.
    
    Args:
        role: ParameterRole
        feature_store: FeatureStore instance
        output_path: Path to save figure
        feature_name: Feature name to plot
    """
    role_features = feature_store.get_by_role(role)
    
    if len(role_features) < 2:
        return
    
    try:
        # Extract feature values and layer indices
        layers = []
        values = []
        
        for tf in sorted(role_features, key=lambda x: x.layer_idx if x.layer_idx is not None else -1):
            if tf.layer_idx is not None and feature_name in tf.features:
                layers.append(tf.layer_idx)
                values.append(tf.features[feature_name])
        
        if len(values) < 2:
            return
        
        # Plot sequence
        plt.figure(figsize=(12, 6))
        plt.plot(layers, values, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel("Layer Index")
        plt.ylabel(feature_name)
        plt.title(f"Sequence Change-Points: {role.value}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

