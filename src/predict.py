import argparse
import json
import torch
import sys
sys.path.append('/content/pii_ner_assignment_IITB/src')

from transformers import AutoModelForTokenClassification, DistilBertTokenizerFast
from labels import id2label, PII_MAP

def align_tokens_to_chars(text, offset_mapping, preds, id2label):
    entities = []
    current_entity = None
    
    for idx, (start, end) in enumerate(offset_mapping):
        if start == end: continue
        
        label_id = preds[idx]
        label_str = id2label[label_id]
        
        if label_str == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
            
        prefix, entity_type = label_str.split("-", 1)
        
        if prefix == "B":
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "start": start, 
                "end": end, 
                "label": entity_type, 
                "pii": PII_MAP.get(entity_type, False)
            }
        elif prefix == "I":
            if current_entity and current_entity["label"] == entity_type:
                current_entity["end"] = end 
            else:
                if current_entity: entities.append(current_entity)
                current_entity = {
                    "start": start, 
                    "end": end, 
                    "label": entity_type, 
                    "pii": PII_MAP.get(entity_type, False)
                }

    if current_entity:
        entities.append(current_entity)
        
    return entities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic quantization")
    args = parser.parse_args()

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    device = torch.device("cpu") 
    model.to(device)
    model.eval()

    print("Applying Dynamic Quantization for Latency Optimization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    results = {}
    
    with open(args.input, 'r') as f:
        lines = f.readlines()
        
    print(f"Predicting on {len(lines)} examples...")
    
    for line in lines:
        data = json.loads(line)
        utt_id = data['id']
        text = data['text']
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True)
        offset_mapping = inputs["offset_mapping"][0].tolist()
        
        del inputs["offset_mapping"]
        
        with torch.no_grad():
            outputs = quantized_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
            
        spans = align_tokens_to_chars(text, offset_mapping, predictions, id2label)
        results[utt_id] = spans

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()