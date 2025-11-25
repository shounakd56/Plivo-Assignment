# PII Entity Recognition for Noisy STT Transcripts

Token-level NER model for detecting PII in noisy Speech-to-Text inputs. Uses a quantized DistilBERT model to achieve <20ms p95 latency on CPU.

## Setup

```bash
pip install -r requirements.txt
python src/generate_data.py

# 1. Train
python src/train.py --model_name distilbert-base-uncased --train data/train.jsonl --dev data/dev.jsonl --out_dir out

# 2. Predict
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json

# 3. Evaluate
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json

# 4. Latency Check
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50