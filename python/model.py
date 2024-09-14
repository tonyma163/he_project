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

#print("x: ", x)

# Save the embedding into text file
path = "./inputs/0"
for i in range(len(x[0])):
    if not (os.path.exists(path)):
        os.makedirs(path)
    np.savetxt("./inputs/0/input_{}.txt".format(i), x[0][i].detach(), delimiter=",")


# Layer 1 - Self Attention
# fetch the weights of Q, K, V
query_weight = model.bert.encoder.layer[0].attention.self.query.weight.clone().detach().double().transpose(0, 1)
key_weight = model.bert.encoder.layer[0].attention.self.key.weight.clone().detach().double().transpose(0, 1)
value_weight = model.bert.encoder.layer[0].attention.self.value.weight.clone().detach().double().transpose(0, 1)

# fetch the biases of Q, K, V
query_bias = model.bert.encoder.layer[0].attention.self.query.bias.clone().detach().double()
key_bias = model.bert.encoder.layer[0].attention.self.key.bias.clone().detach().double()
value_bias = model.bert.encoder.layer[0].attention.self.value.bias.clone().detach().double()

# convert input embeddings into double
input_tensor = x.double()

# calculate the new Q, K, V based on the new input embeddings
query = torch.matmul(input_tensor, query_weight) + query_bias
key = torch.matmul(input_tensor, key_weight) + key_bias
value = torch.matmul(input_tensor, value_weight) + value_bias

# reshape matrics for multi-head attention
query = query.reshape([1, input_tensor.size()[1], 2, 64]) # 2 -> no. of attention heads, 64 -> dimension of each head (hidden size/no. of attention heads)
key = key.reshape([1, input_tensor.size()[1], 2, 64])
value = value.reshape([1, input_tensor.size()[1], 2, 64])

# permute to adjust the dimensions of the tensors for the dot product operations
query = query.permute([0, 2, 1, 3])
key = key.permute([0, 2, 3, 1])

# (Q * K) / square root of dk
qk = torch.matmul(query, key)
qk = qk / 8 # 1 / square root of 64

# Softmax()
qk_softmaxed = torch.softmax(qk, -1)

# permute value
value = value.permute([0, 2, 1, 3])

# 
dot_product = torch.matmul(qk_softmaxed, value)
# permute & reshape for further layer operations
dot_product = dot_product.permute([0, 2, 1, 3])
dot_product = dot_product.reshape([1, input_tensor.size()[1], 128])

print("dot_product: ", dot_product)