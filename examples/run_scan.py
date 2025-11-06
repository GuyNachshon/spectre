#!/usr/bin/env python3
"""Example script to run Spectre scan programmatically."""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectre.core.config import Config
from spectre.core.pipeline import Pipeline
from spectre.core.output import generate_all_outputs


def main():
    """Run Spectre scan."""
    # Configuration
    model_path = Path("/path/to/your/model.safetensors")  # Update this!
    output_dir = Path("./scan_results")
    ruleset = "gpt_like"  # Options: gpt_like, llama_like, mistral_like, phi_like
    
    # Create configuration
    config = Config()
    config.set("model.path", str(model_path))
    config.set("model.ruleset", ruleset)
    config.set("output.dir", str(output_dir))
    config.set("output.save_visuals", True)
    config.set("output.topk_visuals", 20)
    
    # Enable/disable specific detectors
    config.set("svd.enabled", True)
    config.set("interlayer.enabled", True)
    config.set("distribution.enabled", True)
    config.set("robust.enabled", True)
    config.set("energy.enabled", True)
    config.set("rmt.enabled", True)
    config.set("spectrogram.enabled", True)
    config.set("gsp.enabled", True)
    config.set("tda.enabled", True)
    config.set("sequence_cp.enabled", True)
    config.set("ot.enabled", True)
    config.set("multiview.enabled", False)
    config.set("conformal.enabled", True)
    
    # Device configuration
    config.set("device.use_cuda", True)
    config.set("device.cuda_device", 0)
    
    print("Starting Spectre scan...")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Ruleset: {ruleset}")
    print()
    
    # Initialize and run pipeline
    pipeline = Pipeline(config)
    results = pipeline.run(checkpoint_path=model_path)
    
    # Print summary
    summary = results.get("summary", {})
    print()
    print("=" * 60)
    print("Scan Complete!")
    print("=" * 60)
    print(f"Total tensors: {summary.get('total_tensors', 0)}")
    print(f"Ensemble sigma: {summary.get('ensemble_sigma', 0.0):.4f}")
    print(f"Overall flag: {summary.get('flag', 'GREEN')}")
    print()
    
    # Flag distribution
    flag_counts = summary.get("flag_counts", {})
    if flag_counts:
        print("Flag distribution:")
        for flag, count in flag_counts.items():
            print(f"  {flag}: {count}")
        print()
    
    # Top suspects
    suspects = summary.get("suspects", [])
    if suspects:
        print(f"Top {min(5, len(suspects))} suspects:")
        for i, suspect in enumerate(suspects[:5], 1):
            print(f"  {i}. {suspect['name'][:50]}")
            print(f"     Role: {suspect['role']}, Layer: {suspect.get('layer_idx', 'N/A')}")
            print(f"     Score: {suspect['ensemble_score']:.4f}, Flag: {suspect['flag']}")
        print()
    
    print(f"Output directory: {output_dir}")
    print(f"  - fingerprint_v2.json")
    print(f"  - features.csv")
    print(f"  - per_layer_summary.csv")
    if config.get("output.save_visuals", True):
        print(f"  - visualizations/")
    print()


if __name__ == "__main__":
    main()

