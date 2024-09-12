import os
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
import torch
import numpy as np

# Load huggingface access token
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# Load tokenizer & model
logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained("gaunernst/bert-tiny-uncased", token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", token=hf_token)
print(model)

# Testing
model.eval()
text = "Yes Sir!"
text = "[CLS] " + text + " [SEP]"

#tokenized = tokenizer(text)
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

x = model.bert.embeddings(tokens_tensor, torch.tensor([[1] * len(tokenized_text)]))

test = tokenizer(text, return_tensors="pt")

print("author: ", tokens_tensor)
print("author2: ", x)
print("\nnew: ",test)

print(model(tokens_tensor))