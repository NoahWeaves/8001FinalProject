# LLM-Based DDoS Controller Ablation Study

## Setup

```bash
python3 -m venv llm-training-env
source llm-training-env/bin/activate
pip install -r requirements.txt
```

## Prepare Data

Place training data in `llm_training_data/`:
- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

Or regenerate:
```bash
python build_training_dataset.py
```

## Run Experiments

**Zero-shot evaluation:**
```bash
python evaluate_zeroshot.py
```

**Single ablation:**
```bash
# Edit config variables in train_ablation.py first
python train_ablation.py
```

**All ablations:**
```bash
python run_all_ablations.py
```

Results: `ablation_results.csv`
