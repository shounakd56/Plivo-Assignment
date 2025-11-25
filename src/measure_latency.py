import time
import json
import torch
import argparse
import numpy as np
from transformers import AutoModelForTokenClassification, DistilBertTokenizerFast

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()
    
    texts = []
    with open(args.input) as f:
        for line in f:
            texts.append(json.loads(line)['text'])
            if len(texts) >= 50: break
            
    latencies = []    
    _ = model(**tokenizer(texts[0], return_tensors="pt"))
    
    for i in range(args.runs):
        text = texts[i % len(texts)]
        inputs = tokenizer(text, return_tensors="pt")
        
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        end = time.time()
        latencies.append((end - start) * 1000) # ms
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 100)
    print(f"P50: {p50:.2f} ms")
    print(f"P95: {p95:.2f} ms")


if __name__ == "__main__":
    main()