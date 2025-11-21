"""
Zero-Shot and Few-Shot Evaluation of Pretrained LLM
Test out-of-the-box capability before fine-tuning
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from tqdm import tqdm
import re
from collections import defaultdict

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # Fast 1B model
DATA_DIR = "llm_training_data"
TEST_SAMPLES = 500  # Evaluate on 500 test samples
OUTPUT_FILE = "zeroshot_results.json"

print("="*80)
print("Zero-Shot Evaluation: Pretrained Model (No Fine-Tuning)")
print("="*80)


def load_model_and_tokenizer():
    """Load pretrained model without fine-tuning"""
    print("\n[1/2] Loading pretrained model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load with 8-bit quantization for 8GB VRAM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_8bit=True,  # Use 8-bit for inference (faster than 4-bit)
        trust_remote_code=True
    )
    
    print("  ✓ Model loaded (8-bit quantization for 8GB VRAM)")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate model response"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response after "### Response:"
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def parse_response(response):
    """Parse attack type and intensity from response"""
    result = {
        "attack_type": "Unknown",
        "intensity": "Unknown",
        "raw_response": response
    }
    
    # Try to extract JSON
    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result["attack_type"] = parsed.get("attack_type", "Unknown")
            result["intensity"] = parsed.get("intensity", "Unknown")
            return result
        except:
            pass
    
    # Fallback: search for attack types
    attack_keywords = ['UDP', 'TCP', 'SYN', 'DNS', 'HTTP', 'LDAP', 'MSSQL', 
                       'NetBIOS', 'Portmap', 'BENIGN', 'UDPLag']
    for keyword in attack_keywords:
        if keyword.lower() in response.lower():
            result["attack_type"] = keyword
            break
    
    # Search for intensity
    if 'high' in response.lower():
        result["intensity"] = "High"
    elif 'medium' in response.lower():
        result["intensity"] = "Medium"
    elif 'low' in response.lower():
        result["intensity"] = "Low"
    
    return result


def evaluate_zero_shot(model, tokenizer, test_dataset):
    """Evaluate model on test set without fine-tuning"""
    print(f"\n[2/2] Running zero-shot evaluation on {len(test_dataset)} samples...")
    
    results = []
    attack_correct = 0
    intensity_correct = 0
    
    for idx, example in enumerate(tqdm(test_dataset)):
        # Use the prompt from the dataset
        prompt = example['prompt']
        true_attack = example['attack_type']
        true_intensity = example['intensity']
        
        # Generate response
        response = generate_response(model, tokenizer, prompt)
        
        # Parse prediction
        parsed = parse_response(response)
        pred_attack = parsed['attack_type']
        pred_intensity = parsed['intensity']
        
        # Check correctness
        attack_match = (pred_attack.lower() == true_attack.lower())
        intensity_match = (pred_intensity.lower() == true_intensity.lower())
        
        if attack_match:
            attack_correct += 1
        if intensity_match and true_intensity != 'None':
            intensity_correct += 1
        
        results.append({
            'true_attack': true_attack,
            'pred_attack': pred_attack,
            'attack_correct': attack_match,
            'true_intensity': true_intensity,
            'pred_intensity': pred_intensity,
            'intensity_correct': intensity_match,
            'raw_response': parsed['raw_response']
        })
        
        # Show progress every 50 samples
        if (idx + 1) % 50 == 0:
            current_attack_acc = attack_correct / (idx + 1)
            current_intensity_acc = intensity_correct / max(1, sum(1 for r in results if r['true_intensity'] != 'None'))
            print(f"\n  Progress: {idx+1}/{len(test_dataset)}")
            print(f"    Attack accuracy: {current_attack_acc:.2%}")
            print(f"    Intensity accuracy: {current_intensity_acc:.2%}")
    
    return results, attack_correct, intensity_correct


def calculate_metrics(results):
    """Calculate detailed metrics"""
    # Attack type metrics
    attack_correct = sum(1 for r in results if r['attack_correct'])
    attack_accuracy = attack_correct / len(results)
    
    # Intensity metrics (excluding 'None')
    intensity_samples = [r for r in results if r['true_intensity'] != 'None']
    intensity_correct = sum(1 for r in intensity_samples if r['intensity_correct'])
    intensity_accuracy = intensity_correct / len(intensity_samples) if intensity_samples else 0
    
    # Per-class breakdown
    attack_breakdown = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        attack_type = r['true_attack']
        attack_breakdown[attack_type]['total'] += 1
        if r['attack_correct']:
            attack_breakdown[attack_type]['correct'] += 1
    
    return {
        'attack_accuracy': attack_accuracy,
        'intensity_accuracy': intensity_accuracy,
        'total_samples': len(results),
        'attack_correct': attack_correct,
        'intensity_correct': intensity_correct,
        'intensity_samples': len(intensity_samples),
        'per_class': dict(attack_breakdown)
    }


def main():
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Load test data
    print(f"\n📂 Loading test dataset...")
    dataset = load_dataset("json", data_files={"test": f"{DATA_DIR}/test.jsonl"})
    test_dataset = dataset['test'].select(range(min(TEST_SAMPLES, len(dataset['test']))))
    print(f"  ✓ Loaded {len(test_dataset)} test samples")
    
    # Evaluate
    results, attack_correct, intensity_correct = evaluate_zero_shot(
        model, tokenizer, test_dataset
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print("\n" + "="*80)
    print("ZERO-SHOT EVALUATION RESULTS")
    print("="*80)
    print(f"\n📊 Overall Performance:")
    print(f"  Attack Classification Accuracy: {metrics['attack_accuracy']:.2%}")
    print(f"  Intensity Assessment Accuracy: {metrics['intensity_accuracy']:.2%}")
    
    print(f"\n📈 Detailed Metrics:")
    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  Attack correct: {metrics['attack_correct']}/{metrics['total_samples']}")
    print(f"  Intensity correct: {metrics['intensity_correct']}/{metrics['intensity_samples']}")
    
    print(f"\n🎯 Per-Class Attack Accuracy:")
    for attack_type, stats in sorted(metrics['per_class'].items()):
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {attack_type:15s}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save results
    output_data = {
        'model': MODEL_NAME,
        'evaluation_type': 'zero-shot',
        'test_samples': len(test_dataset),
        'metrics': metrics,
        'detailed_results': results[:10]  # Save first 10 for inspection
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {OUTPUT_FILE}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nZero-shot performance shows the pretrained model's capability")
    print("WITHOUT any fine-tuning on your DDoS dataset.")
    print("\nComparison baseline:")
    print(f"  - Your ML baselines (k-NN): ~99% accuracy")
    print(f"  - Zero-shot LLM: {metrics['attack_accuracy']:.1%} accuracy")
    print(f"  - Gap to close with fine-tuning: {(0.99 - metrics['attack_accuracy']):.1%}")
    
    if metrics['attack_accuracy'] < 0.50:
        print("\n💡 Low zero-shot performance suggests fine-tuning is ESSENTIAL")
    elif metrics['attack_accuracy'] < 0.80:
        print("\n💡 Moderate zero-shot performance - fine-tuning should improve significantly")
    else:
        print("\n💡 High zero-shot performance - model already understands task!")
    
    return metrics


if __name__ == "__main__":
    metrics = main()
