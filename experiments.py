"""
Experiments script for research questions

This script runs systematic experiments to answer the research questions:
1. Effect of L1 vs Perceptual Loss
2. Data requirements for quality
3. Multi-style adaptation
4. Pixel-wise vs Patch-wise adversarial loss
"""

import os
import subprocess
import json
from datetime import datetime


class ExperimentRunner:
    """Manages and runs experiments"""
    
    def __init__(self, base_dir="experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.results = []
    
    def run_experiment(self, name, config_overrides, epochs=50):
        """
        Run a single experiment
        
        Args:
            name: Experiment name
            config_overrides: Dictionary of config parameters to override
            epochs: Number of training epochs
        """
        print("\n" + "="*70)
        print(f"Running Experiment: {name}")
        print("="*70)
        print(f"Configuration: {config_overrides}")
        print(f"Epochs: {epochs}")
        
        # Create experiment-specific config
        config_code = f"""
import sys
sys.path.insert(0, '.')
from config import Config

# Override settings
"""
        for key, value in config_overrides.items():
            if isinstance(value, str):
                config_code += f"Config.{key} = '{value}'\n"
            else:
                config_code += f"Config.{key} = {value}\n"
        
        config_code += f"Config.NUM_EPOCHS = {epochs}\n"
        
        # Save experiment config
        exp_dir = os.path.join(self.base_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        
        with open(os.path.join(exp_dir, "config_override.py"), "w") as f:
            f.write(config_code)
        
        # Log experiment
        experiment_info = {
            "name": name,
            "config": config_overrides,
            "epochs": epochs,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        # Run training
        try:
            # Create a temporary training script that loads overrides
            train_script = f"""
import sys
sys.path.insert(0, '.')

# Load config overrides
exec(open('{exp_dir}/config_override.py').read())

# Import and run training
from train import main
main()
"""
            
            script_path = os.path.join(exp_dir, "run_train.py")
            with open(script_path, "w") as f:
                f.write(train_script)
            
            # Run training
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True
            )
            
            experiment_info["status"] = "completed" if result.returncode == 0 else "failed"
            experiment_info["end_time"] = datetime.now().isoformat()
            
            # Save logs
            with open(os.path.join(exp_dir, "stdout.log"), "w") as f:
                f.write(result.stdout)
            with open(os.path.join(exp_dir, "stderr.log"), "w") as f:
                f.write(result.stderr)
            
            print(f"\n{'✓' if result.returncode == 0 else '✗'} Experiment {name} {experiment_info['status']}")
            
        except Exception as e:
            experiment_info["status"] = "error"
            experiment_info["error"] = str(e)
            print(f"✗ Experiment {name} failed with error: {e}")
        
        # Save experiment info
        self.results.append(experiment_info)
        self.save_results()
        
        return experiment_info
    
    def save_results(self):
        """Save all experiment results"""
        results_path = os.path.join(self.base_dir, "experiment_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {results_path}")
    
    def print_summary(self):
        """Print summary of all experiments"""
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        
        for exp in self.results:
            status_symbol = "✓" if exp["status"] == "completed" else "✗"
            print(f"\n{status_symbol} {exp['name']}")
            print(f"   Status: {exp['status']}")
            print(f"   Config: {exp['config']}")


def experiment_1_loss_comparison():
    """
    Research Question 1: Effect of L1 vs Perceptual Loss
    
    Compares three loss configurations:
    - L1 only (traditional)
    - Perceptual only (VGG-based)
    - Combined (both)
    """
    print("\n" + "="*70)
    print("RESEARCH QUESTION 1: L1 vs Perceptual Loss")
    print("="*70)
    
    runner = ExperimentRunner("experiments/loss_comparison")
    
    experiments = [
        ("l1_loss", {"LOSS_TYPE": "l1"}),
        ("perceptual_loss", {"LOSS_TYPE": "perceptual"}),
        ("combined_loss", {"LOSS_TYPE": "combined"}),
    ]
    
    for name, config in experiments:
        runner.run_experiment(name, config, epochs=100)
    
    runner.print_summary()
    
    print("\n" + "="*70)
    print("ANALYSIS RECOMMENDATIONS:")
    print("="*70)
    print("1. Compare visual quality of generated paintings")
    print("2. Check validation L1 loss for each method")
    print("3. Evaluate artistic style adherence")
    print("4. Measure training stability (discriminator loss)")
    print("\nExpected: Combined loss should give best artistic results")


def experiment_2_data_requirements():
    """
    Research Question 2: How much paired data is needed?
    
    Tests different training set sizes:
    - 100 samples
    - 500 samples  
    - 1000 samples
    - All data
    """
    print("\n" + "="*70)
    print("RESEARCH QUESTION 2: Data Requirements")
    print("="*70)
    
    runner = ExperimentRunner("experiments/data_requirements")
    
    experiments = [
        ("data_100", {"TRAIN_SIZE": 100}),
        ("data_500", {"TRAIN_SIZE": 500}),
        ("data_1000", {"TRAIN_SIZE": 1000}),
        ("data_all", {"TRAIN_SIZE": None}),
    ]
    
    for name, config in experiments:
        runner.run_experiment(name, config, epochs=100)
    
    runner.print_summary()
    
    print("\n" + "="*70)
    print("ANALYSIS RECOMMENDATIONS:")
    print("="*70)
    print("1. Plot validation loss vs. dataset size")
    print("2. Identify point of diminishing returns")
    print("3. Check for overfitting in small datasets")
    print("4. Compare visual quality across dataset sizes")
    print("\nExpected: Quality improves rapidly until ~1000 samples, then plateaus")


def experiment_3_discriminator_comparison():
    """
    Research Question 4: Pixel-wise vs Patch-wise adversarial loss
    
    Compares:
    - PatchGAN (70x70 receptive field)
    - PixelGAN (1x1 receptive field)
    """
    print("\n" + "="*70)
    print("RESEARCH QUESTION 4: PatchGAN vs PixelGAN")
    print("="*70)
    
    runner = ExperimentRunner("experiments/discriminator_comparison")
    
    experiments = [
        ("patchgan", {"DISC_TYPE": "patchgan"}),
        ("pixelgan", {"DISC_TYPE": "pixel"}),
    ]
    
    for name, config in experiments:
        runner.run_experiment(name, config, epochs=100)
    
    runner.print_summary()
    
    print("\n" + "="*70)
    print("ANALYSIS RECOMMENDATIONS:")
    print("="*70)
    print("1. Compare texture quality in generated paintings")
    print("2. Check training time for each discriminator")
    print("3. Evaluate spatial coherence and local details")
    print("4. Measure discriminator loss stability")
    print("\nExpected: PatchGAN should produce better texture quality")


def quick_experiment():
    """
    Quick experiment with small dataset and few epochs
    Useful for testing the pipeline
    """
    print("\n" + "="*70)
    print("QUICK TEST EXPERIMENT")
    print("="*70)
    print("Running a quick test with minimal settings...")
    
    runner = ExperimentRunner("experiments/quick_test")
    
    config = {
        "TRAIN_SIZE": 50,
        "BATCH_SIZE": 8,
        "LOSS_TYPE": "combined",
        "DISC_TYPE": "patchgan"
    }
    
    runner.run_experiment("quick_test", config, epochs=10)
    runner.print_summary()
    
    print("\nQuick test complete! Check experiments/quick_test/ for results.")


def full_research_suite():
    """
    Run all research experiments
    WARNING: This will take a long time!
    """
    print("\n" + "="*70)
    print("FULL RESEARCH EXPERIMENT SUITE")
    print("="*70)
    print("\nThis will run all experiments to answer all research questions.")
    print("Estimated time: 8-12 hours on GPU")
    print("\nExperiments to run:")
    print("  1. Loss comparison (3 experiments)")
    print("  2. Data requirements (4 experiments)")
    print("  3. Discriminator comparison (2 experiments)")
    print("\nTotal: 9 experiments × 100 epochs each")
    
    confirm = input("\nProceed? (yes/no): ").strip().lower()
    
    if confirm != "yes":
        print("Cancelled.")
        return
    
    print("\nStarting full research suite...")
    
    # Run all experiments
    experiment_1_loss_comparison()
    experiment_2_data_requirements()
    experiment_3_discriminator_comparison()
    
    print("\n" + "="*70)
    print("FULL RESEARCH SUITE COMPLETE!")
    print("="*70)
    print("\nResults saved in experiments/ directory")
    print("Review experiment_results.json for detailed outcomes")


def analyze_results():
    """
    Analyze and compare experiment results
    """
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    # Look for experiment results
    results_file = "experiments/experiment_results.json"
    
    if not os.path.exists(results_file):
        print("No experiment results found. Run experiments first.")
        return
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    print(f"\nFound {len(results)} completed experiments\n")
    
    # Group by experiment type
    loss_experiments = [r for r in results if "loss" in r["name"]]
    data_experiments = [r for r in results if "data" in r["name"]]
    disc_experiments = [r for r in results if "gan" in r["name"]]
    
    if loss_experiments:
        print("\n--- Loss Comparison Results ---")
        for exp in loss_experiments:
            print(f"  {exp['name']}: {exp['status']}")
    
    if data_experiments:
        print("\n--- Data Requirements Results ---")
        for exp in data_experiments:
            print(f"  {exp['name']}: {exp['status']}")
    
    if disc_experiments:
        print("\n--- Discriminator Comparison Results ---")
        for exp in disc_experiments:
            print(f"  {exp['name']}: {exp['status']}")
    
    print("\n" + "="*70)
    print("For detailed analysis, check:")
    print("  - logs/[experiment_name]/samples/ for visual comparisons")
    print("  - logs/[experiment_name]/plots/ for training curves")
    print("  - checkpoints/[experiment_name]/ for saved models")


def main():
    """Main experiment menu"""
    print("="*70)
    print("AutoPainter Experiment Suite")
    print("="*70)
    print("\nResearch Questions:")
    print("  1. Effect of L1 vs Perceptual Loss")
    print("  2. Data requirements for quality")
    print("  3. Multi-style adaptation (future work)")
    print("  4. Pixel-wise vs Patch-wise adversarial loss")
    print("\n" + "="*70)
    print("Experiment Options:")
    print("="*70)
    print("  1. Quick test (10 epochs, small dataset)")
    print("  2. Loss comparison (L1 vs Perceptual)")
    print("  3. Data requirements study")
    print("  4. Discriminator comparison (PatchGAN vs PixelGAN)")
    print("  5. Run ALL experiments (long!)")
    print("  6. Analyze results")
    print("  0. Exit")
    
    choice = input("\nEnter choice: ").strip()
    
    if choice == "1":
        quick_experiment()
    elif choice == "2":
        experiment_1_loss_comparison()
    elif choice == "3":
        experiment_2_data_requirements()
    elif choice == "4":
        experiment_3_discriminator_comparison()
    elif choice == "5":
        full_research_suite()
    elif choice == "6":
        analyze_results()
    elif choice == "0":
        print("Exiting.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
