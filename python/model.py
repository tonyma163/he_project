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
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny-mnli", token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", token=hf_token)

# Load fine-tuned model weights
trained = torch.load('SST-2-BERT-tiny.bin', map_location=torch.device('cpu'))
model.load_state_dict(trained, strict=False)
print(model)

# Testing
model.eval()

# Input data
text = "Nuovo Cinema Paradiso has been an incredible movie! A gem in the italian culture."
text = "[CLS] " + text + " [SEP]"

# Tokenize input data
#tokenized = tokenizer(text) # <- this would return the input_ids and attention mask as well
tokenized_text = tokenizer.tokenize(text) # <- tokenize the input data
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) # <- convert tokens into ids
tokens_tensor = torch.tensor([indexed_tokens]) # <- convert token ids into tensor

#print("tokenized: ", tokenized)
#print("tokenized_text: ", tokenized_text)
#print("indexed_tokens: ", indexed_tokens)
#print("tokens_tensor: ", tokens_tensor)

# Embedding calculation
x = model.bert.embeddings(tokens_tensor, torch.tensor([[1] * len(tokenized_text)])) # require token ids tensor and attention mask list

print("x: ", x)

# Save the embedding into text file
path = "./inputs/0"
for i in range(len(x[0])):
    if not (os.path.exists(path)):
        os.makedirs(path)
    np.savetxt("./inputs/0/input_{}.txt".format(i), x[0][i].detach(), delimiter=",")
