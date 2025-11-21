"""
Dataset Builder for LLM-based DDoS Controller
- Combines BENIGN and attack samples
- Adds intensity labels using heuristic
- Creates train/val/test splits
- Formats for LLM instruction tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

# Configuration
BENIGN_FILE = "benign_samples_all.csv"
ATTACK_FILE = "attack_samples_5k_per_type.csv"
OUTPUT_DIR = "llm_training_data"

# Important features for LLM (top 20)
IMPORTANT_FEATURES = [
    'Flow Bytes/s',
    'Flow Packets/s',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Packet Length Mean',
    'Flow Duration',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow IAT Mean',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'PSH Flag Count',
    'ACK Flag Count',
    'SYN Flag Count',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Subflow Fwd Bytes'
]


class IntensityLabeler:
    """Calculate attack intensity using heuristic from mid-report"""
    
    def __init__(self, benign_baseline_df):
        """Calculate baseline statistics from benign traffic"""
        self.baseline_stats = self._calculate_baseline(benign_baseline_df)
        
    def _calculate_baseline(self, benign_df):
        """Calculate 95th percentile baseline from benign traffic"""
        features = [
            'Flow Bytes/s', 'Flow Packets/s', 'Total Fwd Packets',
            'Total Backward Packets', 'Total Length of Fwd Packets',
            'Total Length of Bwd Packets', 'Packet Length Mean', 'Flow Duration'
        ]
        
        baseline = {}
        for feat in features:
            if feat in benign_df.columns:
                # Use 95th percentile as baseline threshold
                baseline[feat] = {
                    'mean': benign_df[feat].mean(),
                    'std': benign_df[feat].std(),
                    'p95': benign_df[feat].quantile(0.95),
                    'median': benign_df[feat].median()
                }
        
        print("\nBaseline statistics calculated from BENIGN traffic:")
        for feat, stats in baseline.items():
            print(f"  {feat:30s}: p95={stats['p95']:.2f}")
        
        return baseline
    
    def calculate_intensity_score(self, row):
        """
        Calculate intensity score based on deviation from baseline
        Weighted combination of rate-based, volume-based, and statistical features
        """
        score = 0.0
        weights = {
            # Rate-based features (50% weight)
            'Flow Bytes/s': 0.25,
            'Flow Packets/s': 0.25,
            # Volume-based features (40% weight)
            'Total Fwd Packets': 0.15,
            'Total Backward Packets': 0.10,
            'Total Length of Fwd Packets': 0.10,
            'Total Length of Bwd Packets': 0.05,
            # Statistical features (10% weight)
            'Packet Length Mean': 0.05,
            'Flow Duration': 0.05
        }
        
        for feat, weight in weights.items():
            if feat in self.baseline_stats and feat in row.index:
                baseline_p95 = self.baseline_stats[feat]['p95']
                value = row[feat]
                
                # Calculate how many times above baseline
                if baseline_p95 > 0:
                    deviation = value / baseline_p95
                    score += weight * deviation
        
        return score
    
    def label_intensity(self, df):
        """Add intensity labels to dataframe"""
        print("\nCalculating intensity scores...")
        
        # Calculate scores for all rows
        df['intensity_score'] = df.apply(self.calculate_intensity_score, axis=1)
        
        # Only label intensity for attack samples (BENIGN gets 'None')
        attack_mask = df[' Label'] != 'BENIGN'
        attack_scores = df.loc[attack_mask, 'intensity_score']
        
        # Use quantiles for attack traffic to determine thresholds
        if len(attack_scores) > 0:
            q33 = attack_scores.quantile(0.33)
            q67 = attack_scores.quantile(0.67)
            
            print(f"\nIntensity thresholds (based on attack traffic quantiles):")
            print(f"  Low-Medium boundary: {q33:.2f}")
            print(f"  Medium-High boundary: {q67:.2f}")
            
            # Assign intensity levels
            df['intensity_level'] = 'None'
            df.loc[attack_mask & (df['intensity_score'] < q33), 'intensity_level'] = 'Low'
            df.loc[attack_mask & (df['intensity_score'] >= q33) & (df['intensity_score'] < q67), 'intensity_level'] = 'Medium'
            df.loc[attack_mask & (df['intensity_score'] >= q67), 'intensity_level'] = 'High'
            
            # Show distribution
            print("\nIntensity distribution (attack samples only):")
            print(df[attack_mask]['intensity_level'].value_counts())
        
        return df


class LLMDatasetFormatter:
    """Format dataset for LLM instruction tuning"""
    
    def __init__(self, important_features):
        self.important_features = important_features
    
    def create_prompt(self, row):
        """Create instruction prompt from network flow data"""
        # Extract features
        features = {}
        for feat in self.important_features:
            if feat in row.index:
                value = row[feat]
                if pd.notna(value):
                    # Rename for semantic clarity
                    clean_name = feat.strip().replace(' ', '_').lower()
                    features[clean_name] = float(value)
        
        # Format as JSON for structured input
        features_json = json.dumps(features, indent=2)
        
        prompt = f"""### Network Flow Analysis
You are a network security controller analyzing DDoS attack traffic.

Flow Metrics:
{features_json}

Task: Classify the attack type and assess its intensity level (Low/Medium/High). Provide reasoning for your assessment.

### Response:
"""
        return prompt
    
    def create_completion(self, row):
        """Create expected completion from labels"""
        attack_type = row[' Label'].strip()
        intensity = row['intensity_level']
        
        # Generate reasoning based on attack characteristics
        reasoning = self._generate_reasoning(row, attack_type, intensity)
        
        response = {
            "attack_type": attack_type,
            "intensity": intensity if intensity != 'None' else 'N/A',
            "reasoning": reasoning
        }
        
        return json.dumps(response, indent=2)
    
    def _generate_reasoning(self, row, attack_type, intensity):
        """Generate reasoning based on flow characteristics"""
        reasoning_parts = []
        
        # Rate-based observations
        if 'Flow Packets/s' in row.index:
            pps = row['Flow Packets/s']
            if pps > 10000:
                reasoning_parts.append(f"High packet rate ({pps:.0f} pps) indicates volumetric attack")
            elif pps > 1000:
                reasoning_parts.append(f"Elevated packet rate ({pps:.0f} pps)")
        
        # Bandwidth observations
        if 'Flow Bytes/s' in row.index:
            bps = row['Flow Bytes/s']
            if bps > 1000000:
                reasoning_parts.append(f"High bandwidth consumption ({bps/1000000:.2f} Mbps)")
        
        # Attack-specific patterns
        if attack_type == 'BENIGN':
            reasoning_parts.append("Traffic patterns consistent with normal network behavior")
        elif 'UDP' in attack_type:
            reasoning_parts.append("UDP-based flooding with minimal state overhead")
        elif 'Syn' in attack_type or 'SYN' in attack_type:
            reasoning_parts.append("TCP SYN flood attempting to exhaust connection table")
        elif 'DNS' in attack_type or 'LDAP' in attack_type or 'MSSQL' in attack_type:
            reasoning_parts.append("Amplification attack leveraging protocol reflection")
        elif 'NetBIOS' in attack_type:
            reasoning_parts.append("NetBIOS service exploitation pattern detected")
        elif 'Portmap' in attack_type:
            reasoning_parts.append("Portmap amplification attack characteristics")
        
        # Intensity-based recommendations
        if intensity == 'High':
            reasoning_parts.append("Severity warrants aggressive mitigation")
        elif intensity == 'Medium':
            reasoning_parts.append("Moderate response appropriate")
        elif intensity == 'Low':
            reasoning_parts.append("Conservative response to maintain availability")
        
        return ". ".join(reasoning_parts) + "."
    
    def format_dataset(self, df):
        """Convert dataframe to instruction format"""
        print("\nFormatting dataset for LLM training...")
        
        formatted_examples = []
        for idx, row in df.iterrows():
            prompt = self.create_prompt(row)
            completion = self.create_completion(row)
            
            formatted_examples.append({
                "prompt": prompt,
                "completion": completion,
                "text": prompt + completion,  # Combined for causal LM training
                "attack_type": row[' Label'].strip(),
                "intensity": row['intensity_level']
            })
            
            if idx % 10000 == 0:
                print(f"  Processed {idx:,}/{len(df):,} samples")
        
        return pd.DataFrame(formatted_examples)


def main():
    print("="*80)
    print("LLM Training Dataset Builder")
    print("="*80)
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    benign_df = pd.read_csv(BENIGN_FILE, low_memory=False)
    attack_df = pd.read_csv(ATTACK_FILE, low_memory=False)
    
    print(f"  BENIGN samples: {len(benign_df):,}")
    print(f"  Attack samples: {len(attack_df):,}")
    
    # Step 2: Sample BENIGN to balance with attacks
    print("\n[2/6] Balancing dataset...")
    # Let's use ~35K BENIGN (roughly equal to attacks)
    target_benign = 35000
    if len(benign_df) > target_benign:
        benign_df = benign_df.sample(n=target_benign, random_state=42)
        print(f"  Sampled {target_benign:,} BENIGN samples")
    
    # Step 3: Calculate intensity labels
    print("\n[3/6] Adding intensity labels...")
    labeler = IntensityLabeler(benign_df)
    
    # Label attacks
    attack_df = labeler.label_intensity(attack_df)
    
    # Add 'None' intensity for BENIGN
    benign_df['intensity_score'] = 0.0
    benign_df['intensity_level'] = 'None'
    
    # Step 4: Combine datasets
    print("\n[4/6] Combining datasets...")
    combined_df = pd.concat([benign_df, attack_df], ignore_index=True)
    print(f"  Total samples: {len(combined_df):,}")
    
    # Show distribution
    print("\n  Label distribution:")
    print(combined_df[' Label'].value_counts())
    
    # Step 5: Create train/val/test splits
    print("\n[5/6] Creating train/val/test splits...")
    
    # Stratify by attack type to ensure representation
    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=0.2, 
        random_state=42,
        stratify=combined_df[' Label']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df[' Label']
    )
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    # Step 6: Format for LLM training
    print("\n[6/6] Formatting for LLM training...")
    formatter = LLMDatasetFormatter(IMPORTANT_FEATURES)
    
    train_formatted = formatter.format_dataset(train_df)
    val_formatted = formatter.format_dataset(val_df)
    test_formatted = formatter.format_dataset(test_df)
    
    # Save formatted datasets
    print("\nSaving datasets...")
    train_formatted.to_json(f"{OUTPUT_DIR}/train.jsonl", orient='records', lines=True)
    val_formatted.to_json(f"{OUTPUT_DIR}/val.jsonl", orient='records', lines=True)
    test_formatted.to_json(f"{OUTPUT_DIR}/test.jsonl", orient='records', lines=True)
    
    # Also save as CSV for inspection
    train_df.to_csv(f"{OUTPUT_DIR}/train_raw.csv", index=False)
    val_df.to_csv(f"{OUTPUT_DIR}/val_raw.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test_raw.csv", index=False)
    
    print(f"\n✓ Datasets saved to '{OUTPUT_DIR}/' directory:")
    print(f"  - train.jsonl ({len(train_formatted):,} samples)")
    print(f"  - val.jsonl ({len(val_formatted):,} samples)")
    print(f"  - test.jsonl ({len(test_formatted):,} samples)")
    
    # Show example
    print("\n" + "="*80)
    print("EXAMPLE TRAINING SAMPLE")
    print("="*80)
    print("\nPrompt:")
    print(train_formatted.iloc[0]['prompt'])
    print("\nCompletion:")
    print(train_formatted.iloc[0]['completion'])
    
    print("\n" + "="*80)
    print("READY FOR TRAINING!")
    print("="*80)
    print("\nNext step: Run the LLM training script with these datasets")
    print(f"Training data location: {OUTPUT_DIR}/")
    
    return train_formatted, val_formatted, test_formatted


if __name__ == "__main__":
    train_data, val_data, test_data = main()