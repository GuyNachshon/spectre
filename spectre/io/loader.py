"""Model checkpoint loader for .safetensors, .pt, and .bin formats."""

import os
from pathlib import Path
from typing import Dict, Iterator, Tuple, Union

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load


def load_checkpoint(
    checkpoint_path: Union[str, Path]
) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Load model checkpoint and yield (name, tensor) pairs.
    
    Supports:
    - .safetensors files
    - .pt/.pth PyTorch checkpoints
    - .bin files (PyTorch format)
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Yields:
        (name, tensor) tuples where tensor is numpy array
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    suffix = checkpoint_path.suffix.lower()
    
    if suffix == ".safetensors":
        yield from _load_safetensors(checkpoint_path)
    elif suffix in [".pt", ".pth", ".bin"]:
        yield from _load_pytorch(checkpoint_path)
    else:
        raise ValueError(f"Unsupported checkpoint format: {suffix}")


def _load_safetensors(checkpoint_path: Path) -> Iterator[Tuple[str, np.ndarray]]:
    """Load SafeTensors format."""
    try:
        # Try direct load first (faster for small files)
        tensors = safetensors_load(str(checkpoint_path))
        for name, tensor in tensors.items():
            yield name, tensor.detach().cpu().numpy()
    except Exception:
        # Fall back to streaming for large files
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                yield key, tensor.detach().cpu().numpy()


def _load_pytorch(checkpoint_path: Path) -> Iterator[Tuple[str, np.ndarray]]:
    """Load PyTorch checkpoint format."""
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        
        # Handle different checkpoint structures
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError(f"Unexpected checkpoint structure: {type(checkpoint)}")
        
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                yield name, tensor.detach().cpu().numpy()
            else:
                # Skip non-tensor values
                continue
                
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch checkpoint: {e}")


def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Dict:
    """
    Get basic information about a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    info = {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "format": checkpoint_path.suffix.lower(),
        "num_tensors": 0,
        "total_params": 0,
    }
    
    tensor_shapes = {}
    for name, tensor in load_checkpoint(checkpoint_path):
        info["num_tensors"] += 1
        info["total_params"] += tensor.size
        tensor_shapes[name] = list(tensor.shape)
    
    info["tensor_shapes"] = tensor_shapes
    return info

