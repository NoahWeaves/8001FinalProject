import ollama
import pandas as pd
import json
from pathlib import Path
import time

class LocalLLMController:
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        print(f"Initialized controller with {model_name}")
    
    def create_prompt(self, flow_features):
        """Simplified, directive prompt"""
        
        prompt = f"""You are a network security analyzer. Classify this DDoS attack.

    FLOW DATA:
    Packet Size: {flow_features.get('Packet Length Mean', 0):.0f} bytes
    Forward Packets: {flow_features.get('Total Fwd Packets', 0):.0f}
    Backward Packets: {flow_features.get('Total Backward Packets', 0):.0f}
    Packets/sec: {flow_features.get('Flow Packets/s', 0):.0f}

    RULES:
    - Packet size 6-40 bytes + Backward packets > 0 → SYN
    - Packet size > 1000 bytes + Backward packets = 0 → LDAP or MSSQL
    - Packet size 300-700 bytes + Backward packets = 0 → UDP or Portmap
    - Packet size 200-400 bytes + Backward packets = 0 → NetBIOS
    - Packets/sec < 100 + Backward packets > 0 → BENIGN

    You MUST respond in EXACTLY this format (no extra text):

    ATTACK_TYPE: [choose one: BENIGN, SYN, UDP, MSSQL, LDAP, NetBIOS, Portmap, UDPLag]
    INTENSITY: [choose one: Low, Medium, High]
    POLICY: [choose one: aggressive, moderate, conservative]
    REASONING: [one brief sentence]
    """
        return prompt
    
    def parse_response(self, response_text):
        """Extract structured data from LLM response"""
        
        result = {
            'attack_type': None,
            'intensity': None,
            'policy': None,
            'reasoning': None,
            'raw_response': response_text
        }
        
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith('ATTACK_TYPE:') or line.upper().startswith('ATTACK TYPE:'):
                result['attack_type'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('INTENSITY:'):
                result['intensity'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('POLICY:') or line.upper().startswith('RECOMMENDED POLICY:'):
                result['policy'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
        
        return result
    
    def classify(self, flow_features):
        """Main classification method"""
        
        prompt = self.create_prompt(flow_features)
        
        try:
            # Call Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 500,  # INCREASE FROM 200 to 500
                }
            )
            
            response_text = response['response']
            parsed = self.parse_response(response_text)
            
            return parsed
        except Exception as e:
            print(f"Error during classification: {e}")
            return {
                'attack_type': None,
                'intensity': None,
                'policy': None,
                'reasoning': None,
                'raw_response': str(e)
            }


def test_on_sample_data(csv_path='combined_sample_10k.csv', n_samples=5):
    """Test the controller on a few samples from your dataset"""
    
    # Load your CIC-DDoS2019 data
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows")
        print(f"✓ Columns found: {len(df.columns)}")
        print(f"\nFirst few column names: {list(df.columns[:10])}")
        
        # Check if Label column exists
        if 'Label' not in df.columns:
            print("\n⚠ Warning: 'Label' column not found. Available columns:")
            print(df.columns.tolist())
            return
            
        # Show label distribution
        print(f"\nLabel distribution in dataset:")
        print(df['Label'].value_counts())
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_path}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Please ensure the CSV file is in the current directory or provide full path")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Take a stratified sample to get diverse attack types
    print(f"\nSampling {n_samples} examples...")
    if len(df) < n_samples:
        test_samples = df
    else:
        # Try to get diverse samples
        try:
            test_samples = df.groupby('Label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, n_samples // df['Label'].nunique())))
            ).sample(n=min(n_samples, len(df)))
        except:
            test_samples = df.sample(n=n_samples, random_state=42)
    
    # Initialize controller
    print(f"\nInitializing LLM controller...")
    controller = LocalLLMController(model_name="llama3.2:3b")
    
    print("\n" + "="*80)
    print("Testing LLM Controller on Sample Data")
    print("="*80 + "\n")
    
    results = []
    start_time = time.time()
    
    for i, (idx, row) in enumerate(test_samples.iterrows()):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{len(test_samples)} (Row {idx})")
        print(f"{'='*80}")
        print(f"Ground Truth Label: {row['Label']}")
        
        # In the test_on_sample_data function, update flow_features extraction:
        flow_features = {
            'Flow Bytes/s': row.get('Flow Bytes/s', 0),
            'Flow Packets/s': row.get('Flow Packets/s', 0),
            'Total Fwd Packets': row.get('Total Fwd Packets', 0),
            'Total Backward Packets': row.get('Total Backward Packets', 0),
            'Packet Length Mean': row.get('Packet Length Mean', 0),
            'Flow Duration': row.get('Flow Duration', 0),
            'Average Packet Size': row.get('Average Packet Size', 0),
            'Down/Up Ratio': row.get('Down/Up Ratio', 0),
            'Destination Port': row.get('Destination Port', 0),  # ADD THIS
            'Protocol': row.get('Protocol', 0),  # ADD THIS
        }
        
        # Show features being used
        print(f"\nFeatures:")
        for key, value in flow_features.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Get LLM prediction
        print(f"\n⏳ Querying LLM...")
        prediction = controller.classify(flow_features)
        
        print(f"\n  RAW RESPONSE:\n{prediction['raw_response']}")

        
        # Check if classification matches
        match = "✓ MATCH" if prediction['attack_type'] and prediction['attack_type'].upper() in str(row['Label']).upper() else "✗ MISMATCH"
        print(f"\n  {match}")
        
        # Store results
        results.append({
            'row_index': idx,
            'ground_truth': row['Label'],
            'predicted_type': prediction['attack_type'],
            'predicted_intensity': prediction['intensity'],
            'recommended_policy': prediction['policy'],
            'reasoning': prediction['reasoning'],
            'flow_bytes_per_s': flow_features['Flow Bytes/s'],
            'flow_packets_per_s': flow_features['Flow Packets/s'],
        })
        
        print(f"\n{'-'*80}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Completed {len(results)} samples in {elapsed:.2f} seconds ({elapsed/len(results):.2f} s/sample)")
    print(f"{'='*80}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = 'llm_controller_test_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to '{output_file}'")
    
    # Quick accuracy check
    if len(results) > 0:
        correct = sum(1 for r in results if r['predicted_type'] and 
                     r['predicted_type'].upper() in str(r['ground_truth']).upper())
        print(f"\nQuick accuracy check: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    
    return results


def test_single_example():
    """Test with a single hardcoded example to verify setup"""
    
    print("="*80)
    print("Testing with a single hardcoded example...")
    print("="*80)
    
    controller = LocalLLMController(model_name="llama3.2:3b")
    
    # Example of a high-intensity UDP attack
    sample_features = {
        'Flow Bytes/s': 2500000,  # High bytes/sec
        'Flow Packets/s': 50000,   # High packets/sec
        'Total Fwd Packets': 1000,
        'Total Backward Packets': 50,
        'Packet Length Mean': 500,
        'Flow Duration': 1000000,
        'Average Packet Size': 475,
        'Down/Up Ratio': 0.05,
    }
    
    print("\nInput Features:")
    for key, value in sample_features.items():
        print(f"  {key}: {value}")
    
    print("\n⏳ Querying LLM...")
    prediction = controller.classify(sample_features)
    
    print("\n" + "="*80)
    print("LLM Response:")
    print("="*80)
    print(f"\nAttack Type: {prediction['attack_type']}")
    print(f"Intensity: {prediction['intensity']}")
    print(f"Policy: {prediction['policy']}")
    print(f"Reasoning: {prediction['reasoning']}")
    print(f"\n--- Raw Response ---")
    print(prediction['raw_response'])
    print("="*80)


if __name__ == "__main__":
    import sys
    
    # Start with single example test
    print("\n" + "="*80)
    print("SINGLE EXAMPLE TEST")
    print("="*80)
    test_single_example()
    
    # Ask if user wants to continue
    print("\n" + "="*80)
    response = input("\nContinue with dataset test? (y/n): ")
    
    if response.lower() == 'y':
        print("\n" + "="*80)
        print("DATASET TEST")
        print("="*80)
        
        # You can adjust n_samples here (start small!)
        test_on_sample_data(csv_path='combined_sample_10k.csv', n_samples=5)
    else:
        print("\nSkipping dataset test. Run again and choose 'y' when ready!")