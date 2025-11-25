import json
import torch
from torch.utils.data import Dataset
from labels import label2id

class PIIDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entities = item.get('entities', [])

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        labels = [label2id["O"]] * len(encoding["input_ids"][0])        
        offsets = encoding["offset_mapping"][0].tolist()
        
        for idx, (start, end) in enumerate(offsets):
            if start == end: continue
            
            token_label = "O"
            for ent in entities:
                estart, eend, elabel = ent['start'], ent['end'], ent['label']
                
                if start >= estart and end <= eend:
                    if start == estart:
                        token_label = f"B-{elabel}"
                    else:
                        token_label = f"I-{elabel}"
                    break
            
            labels[idx] = label2id.get(token_label, 0)

        item_dict = {key: val.squeeze() for key, val in encoding.items() if key != 'offset_mapping'}
        item_dict['labels'] = torch.tensor(labels)
        return item_dict