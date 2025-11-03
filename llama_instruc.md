# LLM-Based DDoS Attack Classifier

Uses a local LLM (via Ollama) to classify network traffic as benign or various DDoS attack types (SYN, UDP, MSSQL, LDAP, NetBIOS, Portmap, UDPLag).

## Setup

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.2:3b

# Install Python deps
pip install ollama pandas
```

## Data Format

CSV with these columns: `Label`, `Flow Bytes/s`, `Flow Packets/s`, `Total Fwd Packets`, `Total Backward Packets`, `Packet Length Mean`, `Flow Duration`, `Average Packet Size`, `Down/Up Ratio`, `Destination Port`, `Protocol`

Default filename: `combined_sample_10k.csv`

## Usage

```bash
ollama serve  # if not running
python llm_ddos_classifier.py
```

Runs single test first, then prompts for dataset test (default 5 samples).

**Change sample count:**
```python
test_on_sample_data(csv_path='combined_sample_10k.csv', n_samples=20)
```

## Output

- Console: features, predictions, match/mismatch, accuracy
- CSV: `llm_controller_test_results.csv` with predictions and reasoning

## Performance

~5-10 sec/sample. Accuracy varies (typically 40-70%) depending on dataset.

## Troubleshooting

- Model error: `ollama pull llama3.2:3b`
- Connection error: `ollama serve`
- Different model: Change `model_name="llama3.2:3b"` in script