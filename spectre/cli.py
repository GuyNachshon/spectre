"""Command-line interface for Spectre."""

import argparse
import sys
from pathlib import Path

from spectre.core.config import Config
from spectre.core.output import generate_all_outputs
from spectre.core.pipeline import Pipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spectre: Weight Outlier Detection & Integrity Fingerprint Engine"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scan_config.yaml"),
        help="Path to configuration file (default: scan_config.yaml)"
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to model checkpoint (overrides config)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--ruleset",
        type=str,
        choices=["gpt_like", "llama_like", "mistral_like", "phi_like"],
        help="Model ruleset (overrides config)"
    )
    
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Skip visualization generation"
    )
    
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of top suspects for visualizations (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config.exists():
        config = Config(config_path=args.config)
    else:
        print(f"Warning: Config file not found: {args.config}. Using defaults.")
        config = Config()
    
    # Override config with command-line arguments
    if args.model:
        config.set("model.path", str(args.model))
    
    if args.output:
        config.set("output.dir", str(args.output))
    
    if args.ruleset:
        config.set("model.ruleset", args.ruleset)
    
    if args.no_visuals:
        config.set("output.save_visuals", False)
    
    if args.topk:
        config.set("output.topk_visuals", args.topk)
    
    # Validate configuration
    model_path = config.get("model.path")
    if model_path is None:
        print("Error: Model path not specified. Use --model or set in config.")
        sys.exit(1)
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    # Get output directory
    output_dir = Path(config.get("output.dir", "./scan_out"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Spectre scan...")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Ruleset: {config.get('model.ruleset', 'gpt_like')}")
    print()
    
    try:
        # Initialize pipeline
        pipeline = Pipeline(config)
        
        # Run pipeline
        print("Running pipeline...")
        results = pipeline.run(checkpoint_path=model_path)
        
        # Generate outputs
        print("Generating outputs...")
        save_visuals = config.get("output.save_visuals", True) and not args.no_visuals
        topk_visuals = config.get("output.topk_visuals", 20)
        
        generate_all_outputs(
            pipeline,
            output_dir,
            save_visuals=save_visuals,
            topk_visuals=topk_visuals
        )
        
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
        
        flag_counts = summary.get("flag_counts", {})
        if flag_counts:
            print("Flag distribution:")
            for flag, count in flag_counts.items():
                print(f"  {flag}: {count}")
            print()
        
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
        if save_visuals:
            print(f"  - visualizations/")
        print()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

