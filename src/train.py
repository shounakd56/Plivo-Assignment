import argparse
import sys
import os
sys.path.append('/content/pii_ner_assignment_IITB/src')

from transformers import (
    AutoModelForTokenClassification, 
    DistilBertTokenizerFast, 
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification
)
from dataset import PIIDataset
from labels import LABEL_LIST, id2label, label2id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    train_dataset = PIIDataset(args.train, tokenizer)
    dev_dataset = PIIDataset(args.dev, tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    # Training Args
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=5e-5
    )



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    print("Starting Training...")
    trainer.train()
    
    print(f"Saving model to {args.out_dir}")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

if __name__ == "__main__":
    main()