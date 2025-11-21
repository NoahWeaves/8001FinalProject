"""
Master Ablation Script - Run All Experiments
Automatically runs 9 ablation experiments and compiles results
"""

import os
import sys
import json
import time
from datetime import datetime
import pandas as pd
import subprocess

# Ablation configurations
ABLATIONS = [
    # Phase 1: Training amount
    {
        "name": "1k_3epochs",
        "samples": 1000,
        "epochs": 3,
        "features": 20,
        "lora_r": 16,
        "max_length": 1024,
        "description": "Baseline: 1K samples, 3 epochs"
    },
    {
        "name": "3k_3epochs",
        "samples": 3000,
        "epochs": 3,
        "features": 20,
        "lora_r": 16,
        "max_length": 1024,
        "description": "More data: 3K samples (ceiling test)"
    },
    
    # Phase 2: Model capacity
    {
        "name": "lora_r8",
        "samples": 1000,
        "epochs": 3,
        "features": 20,
        "lora_r": 8,
        "max_length": 1024,
        "description": "Lower LoRA rank: r=8"
    },
    {
        "name": "lora_r4",
        "samples": 1000,
        "epochs": 3,
        "features": 20,
        "lora_r": 4,
        "max_length": 1024,
        "description": "Minimal LoRA rank: r=4"
    },
    
    # Phase 3: Feature reduction
    {
        "name": "10features",
        "samples": 1000,
        "epochs": 3,
        "features": 10,
        "lora_r": 16,
        "max_length": 1024,
        "description": "Feature reduction: 10 features"
    },
    {
        "name": "5features",
        "samples": 1000,
        "epochs": 3,
        "features": 5,
        "lora_r": 16,
        "max_length": 1024,
        "description": "Minimal features: 5 features"
    },
    
    # Phase 4: Sequence length
    {
        "name": "seq512",
        "samples": 1000,
        "epochs": 3,
        "features": 20,
        "lora_r": 16,
        "max_length": 512,
        "description": "Shorter sequence: 512 tokens"
    },
    {
        "name": "seq256",
        "samples": 1000,
        "epochs": 3,
        "features": 20,
        "lora_r": 16,
        "max_length": 256,
        "description": "Minimal sequence: 256 tokens"
    },
]

# Reference baseline (already completed)
BASELINE = {
    "name": "baseline_with_labels_1epoch",
    "samples": 1000,
    "epochs": 1,
    "features": 20,
    "lora_r": 16,
    "max_length": 1024,
    "description": "Original baseline: 1 epoch",
    "attack_acc": 0.49,  # From previous run
    "intensity_acc": 0.33
}

def print_header():
    """Print experiment header"""
    print("\n" + "="*80)
    print("MASTER ABLATION STUDY")
    print("="*80)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTotal experiments: {len(ABLATIONS)}")
    print("\nExperiment plan:")
    for i, config in enumerate(ABLATIONS, 1):
        print(f"  {i}. {config['name']:20s} - {config['description']}")
    print("\n" + "="*80)
    
    response = input("\nProceed with all experiments? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)


def run_training(config):
    """Run training for a single ablation"""
    print("\n" + "="*80)
    print(f"Training: {config['name']}")
    print("="*80)
    print(f"  Description: {config['description']}")
    
    start_time = time.time()
    
    try:
        # Run parameterized training script
        cmd = [
            'python', 'train_ablation_parameterized.py',
            '--name', config['name'],
            '--samples', str(config['samples']),
            '--epochs', str(config['epochs']),
            '--lora_r', str(config['lora_r']),
            '--max_length', str(config['max_length']),
            '--features', str(config['features'])
        ]
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"  ✓ Training completed in {elapsed/60:.1f} minutes")
            return True, elapsed
        else:
            print(f"  ✗ Training failed!")
            return False, elapsed
            
    except Exception as e:
        print(f"  ✗ Training crashed: {e}")
        return False, time.time() - start_time


def run_evaluation(config):
    """Run evaluation for a single ablation"""
    print(f"\n  Evaluating {config['name']}...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', 'evaluate_trained_model.py', f"ablation_{config['name']}"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Parse results
            results_file = f"ablation_{config['name']}/evaluation_results.json"
            if os.path.exists(results_file):
                with open(results_file) as f:
                    data = json.load(f)
                    attack_acc = data['metrics']['attack_accuracy']
                    intensity_acc = data['metrics']['intensity_accuracy']
                    
                print(f"  ✓ Evaluation completed in {elapsed/60:.1f} minutes")
                print(f"    Attack: {attack_acc:.1%}, Intensity: {intensity_acc:.1%}")
                return True, attack_acc, intensity_acc, elapsed
            else:
                print(f"  ✗ Results file not found")
                return False, 0.0, 0.0, elapsed
        else:
            print(f"  ✗ Evaluation failed")
            return False, 0.0, 0.0, elapsed
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Evaluation timeout (30 min)")
        return False, 0.0, 0.0, 1800
    except Exception as e:
        print(f"  ✗ Evaluation crashed: {e}")
        return False, 0.0, 0.0, time.time() - start_time


def compile_results(results):
    """Compile all results into summary"""
    print("\n" + "="*80)
    print("COMPILING RESULTS")
    print("="*80)
    
    # Create DataFrame
    df_data = []
    
    # Add baseline
    df_data.append({
        'Experiment': BASELINE['name'],
        'Samples': BASELINE['samples'],
        'Epochs': BASELINE['epochs'],
        'LoRA_r': BASELINE['lora_r'],
        'Features': BASELINE['features'],
        'Max_Length': BASELINE['max_length'],
        'Attack_Acc': BASELINE['attack_acc'],
        'Intensity_Acc': BASELINE['intensity_acc'],
        'Train_Time_min': 15,  # Approximate
        'Eval_Time_min': 26,
        'Total_Time_min': 41,
        'Description': BASELINE['description']
    })
    
    # Add new results
    for result in results:
        df_data.append({
            'Experiment': result['name'],
            'Samples': result['samples'],
            'Epochs': result['epochs'],
            'LoRA_r': result['lora_r'],
            'Features': result['features'],
            'Max_Length': result['max_length'],
            'Attack_Acc': result['attack_acc'],
            'Intensity_Acc': result['intensity_acc'],
            'Train_Time_min': result['train_time'] / 60,
            'Eval_Time_min': result['eval_time'] / 60,
            'Total_Time_min': (result['train_time'] + result['eval_time']) / 60,
            'Description': result['description']
        })
    
    df = pd.DataFrame(df_data)
    
    # Save CSV
    df.to_csv('ablation_results_summary.csv', index=False)
    print("\n  ✓ Saved to: ablation_results_summary.csv")
    
    # Print table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print("\n" + df.to_string(index=False))
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    best_acc = df.loc[df['Attack_Acc'].idxmax()]
    print(f"\n🏆 Best Attack Accuracy: {best_acc['Experiment']}")
    print(f"   {best_acc['Attack_Acc']:.1%} ({best_acc['Description']})")
    
    fastest = df.loc[df['Total_Time_min'].idxmin()]
    print(f"\n⚡ Fastest Training: {fastest['Experiment']}")
    print(f"   {fastest['Total_Time_min']:.1f} min ({fastest['Description']})")
    
    # Insights
    print("\n📊 Insights:")
    
    # Training amount
    baseline_1k = df[df['Experiment'] == '1k_3epochs']['Attack_Acc'].values[0]
    ceiling_3k = df[df['Experiment'] == '3k_3epochs']['Attack_Acc'].values[0]
    print(f"  • More data: {baseline_1k:.1%} (1K) → {ceiling_3k:.1%} (3K) = +{(ceiling_3k-baseline_1k)*100:.1f}%")
    
    # LoRA rank
    r16_acc = df[df['Experiment'] == '1k_3epochs']['Attack_Acc'].values[0]
    r8_acc = df[df['Experiment'] == 'lora_r8']['Attack_Acc'].values[0]
    r4_acc = df[df['Experiment'] == 'lora_r4']['Attack_Acc'].values[0]
    print(f"  • LoRA rank: r=16 ({r16_acc:.1%}), r=8 ({r8_acc:.1%}), r=4 ({r4_acc:.1%})")
    
    # Features
    f20_acc = df[df['Experiment'] == '1k_3epochs']['Attack_Acc'].values[0]
    f10_acc = df[df['Experiment'] == '10features']['Attack_Acc'].values[0]
    f5_acc = df[df['Experiment'] == '5features']['Attack_Acc'].values[0]
    print(f"  • Features: 20 ({f20_acc:.1%}), 10 ({f10_acc:.1%}), 5 ({f5_acc:.1%})")
    
    # Sequence length
    s1024_acc = df[df['Experiment'] == '1k_3epochs']['Attack_Acc'].values[0]
    s512_acc = df[df['Experiment'] == 'seq512']['Attack_Acc'].values[0]
    s256_acc = df[df['Experiment'] == 'seq256']['Attack_Acc'].values[0]
    print(f"  • Seq length: 1024 ({s1024_acc:.1%}), 512 ({s512_acc:.1%}), 256 ({s256_acc:.1%})")
    
    return df


def main():
    """Main execution"""
    print_header()
    
    start_time = time.time()
    results = []
    
    for i, config in enumerate(ABLATIONS, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(ABLATIONS)}")
        print(f"{'='*80}")
        
        # Train
        train_success, train_time = run_training(config)
        
        if not train_success:
            print(f"  ⚠️  Skipping evaluation due to training failure")
            continue
        
        # Evaluate
        eval_success, attack_acc, intensity_acc, eval_time = run_evaluation(config)
        
        # Record results
        results.append({
            'name': config['name'],
            'samples': config['samples'],
            'epochs': config['epochs'],
            'lora_r': config['lora_r'],
            'features': config['features'],
            'max_length': config['max_length'],
            'description': config['description'],
            'train_success': train_success,
            'eval_success': eval_success,
            'attack_acc': attack_acc,
            'intensity_acc': intensity_acc,
            'train_time': train_time,
            'eval_time': eval_time
        })
        
        # Progress update
        completed = i
        remaining = len(ABLATIONS) - i
        elapsed = time.time() - start_time
        avg_time = elapsed / completed
        est_remaining = avg_time * remaining
        
        print(f"\n  Progress: {completed}/{len(ABLATIONS)} complete")
        print(f"  Elapsed: {elapsed/3600:.1f}h, Estimated remaining: {est_remaining/3600:.1f}h")
    
    # Compile final results
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {total_time/3600:.2f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate summary
    df = compile_results(results)
    
    print("\n" + "="*80)
    print("✅ ABLATION STUDY COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review ablation_results_summary.csv")
    print("  2. Create visualizations for paper")
    print("  3. Write analysis section")
    
    return df


if __name__ == "__main__":
    try:
        df = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Partial results may be available in ablation_results_summary.csv")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
