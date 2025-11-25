LABEL_LIST = [
    "O",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-DATE", "I-DATE",
    "B-CITY", "I-CITY",
    "B-LOCATION", "I-LOCATION",
]

id2label = {i: label for i, label in enumerate(LABEL_LIST)}
label2id = {label: i for i, label in enumerate(LABEL_LIST)}

PII_MAP = {
    "CREDIT_CARD": True,
    "PHONE": True,
    "EMAIL": True,
    "PERSON_NAME": True,
    "DATE": True,
    "CITY": False,
    "LOCATION": False
}