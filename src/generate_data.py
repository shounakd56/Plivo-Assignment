import json
import random
from faker import Faker
from tqdm import tqdm

fake = Faker()

LABELS = ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"]

def noise_email(text):
    return text.replace("@", " at ").replace(".", " dot ").lower()

def noise_phone(text):
    text = text.replace("-", " ").replace("(", "").replace(")", "").replace(".", "")
    mapping = {'0': ' zero ', '1': ' one ', '2': ' two '}
    if random.random() < 0.3:
        text = "".join([mapping.get(c, c) for c in text])
    return " ".join(text.split())

def noise_card(text):
    return text.replace(" ", "").replace("-", " ")

def generate_example(id_num):
    templates = [
        "my name is {PERSON_NAME} and i live in {CITY}",
        "please contact me at {EMAIL}",
        "record the date {DATE} for the meeting",
        "my phone number is {PHONE}",
        "i used my credit card {CREDIT_CARD} to buy it",
        "meeting at {LOCATION} on {DATE}",
        "send the file to {PERSON_NAME} at {EMAIL}",
        "is {CITY} located in {LOCATION}",
        "my details are {PERSON_NAME} {PHONE} and {EMAIL}"
    ]
    
    template = random.choice(templates)
    
    entities = []
    text_parts = []
    current_len = 0

    raw_tokens = template.split()
    
    final_text = ""
    
    for token in raw_tokens:
        clean_token = token.replace("{", "").replace("}", "")
        
        if "{" in token:
            label = clean_token
            val = ""
            
            if label == "PERSON_NAME": val = fake.name()
            elif label == "CITY": val = fake.city()
            elif label == "LOCATION": val = fake.country()
            elif label == "DATE": val = fake.date()
            elif label == "EMAIL": val = noise_email(fake.email())
            elif label == "PHONE": val = noise_phone(fake.phone_number())
            elif label == "CREDIT_CARD": val = noise_card(fake.credit_card_number())
            
            val = val.lower()
            
            start = len(final_text)
            final_text += val
            end = len(final_text)
            
            entities.append({"start": start, "end": end, "label": label})
        else:
            final_text += token
            
        final_text += " " # Add space after token
        
    final_text = final_text.strip()
    
    return {
        "id": f"utt_{id_num:04d}",
        "text": final_text,
        "entities": entities
    }

def save_jsonl(data, filepath):
    with open(filepath, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

print("Generating Data...")
train_data = [generate_example(i) for i in range(1000)]
dev_data = [generate_example(i) for i in range(1000, 1200)]
stress_data = [generate_example(i) for i in range(2000, 2100)]

save_jsonl(train_data, '/content/pii_ner_assignment_IITB/data/train.jsonl')
save_jsonl(dev_data, '/content/pii_ner_assignment_IITB/data/dev.jsonl')
save_jsonl(stress_data, '/content/pii_ner_assignment_IITB/data/stress.jsonl')
print("Data Generation Complete.")