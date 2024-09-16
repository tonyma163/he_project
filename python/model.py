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

#type = "./precomputed"
type = "./precomputed2"
#type = "./precomputed3"
#type = "./precomputed4"
#type = "./precomputed5"
# Load precomputed means and variances
mean_0_0 = np.loadtxt(f"{type}/layer0_self_output_mean.txt")
inv_sqrt_var_0_0 = np.loadtxt(f"{type}/layer0_self_output_inv_sqrt_var.txt")

mean_0_1 = np.loadtxt(f"{type}/layer0_output_mean.txt")
inv_sqrt_var_0_1 = np.loadtxt(f"{type}/layer0_output_inv_sqrt_var.txt")

mean_1_0 = np.loadtxt(f"{type}/layer1_self_output_mean.txt")
inv_sqrt_var_1_0 = np.loadtxt(f"{type}/layer1_self_output_inv_sqrt_var.txt")

mean_1_1 = np.loadtxt(f"{type}/layer1_output_mean.txt")
inv_sqrt_var_1_1 = np.loadtxt(f"{type}/layer1_output_inv_sqrt_var.txt")

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


# Layer 0 - Self Attention
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

# dot product
fin = torch.matmul(qk_softmaxed, value)
# permute & reshape for further layer operations
fin = fin.permute([0, 2, 1, 3])
fin = fin.reshape([1, input_tensor.size()[1], 128])

print("dot_product: ", fin)

# Layer 0 - Self Output
weight_output_dense = model.bert.encoder.layer[0].attention.output.dense.weight.clone().detach().double().transpose(0, 1)
bias_output_dense = model.bert.encoder.layer[0].attention.output.dense.bias.clone().detach().double()

fin2 = torch.matmul(fin, weight_output_dense) + bias_output_dense
fin2_backup = fin2.clone()
fin2_backup = fin2_backup + input_tensor

fin3_whole = []

for i in range(len(input_tensor.squeeze())):
    fin2 = fin2_backup.squeeze()[i]
    fin3_corr = (fin2.squeeze().detach() - mean_0_0[i]) * inv_sqrt_var_0_0[i]

    weight_output_layernorm = model.bert.encoder.layer[0].attention.output.LayerNorm.weight.clone().detach().double().unsqueeze(0)
    bias_output_layernorm = model.bert.encoder.layer[0].attention.output.LayerNorm.bias.clone().detach().double()

    fin3_corr = fin3_corr * weight_output_layernorm + bias_output_layernorm
    fin3_whole.append(fin3_corr.detach())

fin3_whole = torch.cat(tuple(fin3_whole), 0).unsqueeze(0)

# Layer 0 - Intermediate
fin_4 = torch.matmul(fin3_whole, model.bert.encoder.layer[0].intermediate.dense.weight.transpose(0, 1).double())
+ model.bert.encoder.layer[0].intermediate.dense.bias
fin_5 = torch.nn.functional.gelu(fin_4)

# Layer 0 - Output
fin_6 = torch.matmul(fin_5, model.bert.encoder.layer[0].output.dense.weight.transpose(0, 1).double())
+ model.bert.encoder.layer[0].output.dense.bias

fin7_whole = []
for i in range(len(input_tensor.squeeze())):
    fin_7 = fin_6.squeeze()[i]

    fin7_corr = (fin_7.squeeze().detach() - mean_0_1[i]) * inv_sqrt_var_0_1[i]

    weight_output_layernorm = model.bert.encoder.layer[0].attention.output.LayerNorm.weight.clone().detach().double().unsqueeze(0)
    bias_output_layernorm = model.bert.encoder.layer[0].attention.output.LayerNorm.bias.clone().detach().double()

    fin7_corr = fin7_corr * weight_output_layernorm + bias_output_layernorm
    fin7_whole.append(fin7_corr.detach())

fin7_whole = torch.cat(tuple(fin7_whole), 0).unsqueeze(0)

# Layer 1 - Self Attention
# fetch the weights of Q, K, V
query_weight = model.bert.encoder.layer[1].attention.self.query.weight.clone().detach().double().transpose(0, 1)
key_weight = model.bert.encoder.layer[1].attention.self.key.weight.clone().detach().double().transpose(0, 1)
value_weight = model.bert.encoder.layer[1].attention.self.value.weight.clone().detach().double().transpose(0, 1)

# fetch the biases of Q, K, V
query_bias = model.bert.encoder.layer[1].attention.self.query.bias.clone().detach().double()
key_bias = model.bert.encoder.layer[1].attention.self.key.bias.clone().detach().double()
value_bias = model.bert.encoder.layer[1].attention.self.value.bias.clone().detach().double()

#
input_tensor = fin7_whole

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

# dot product
fin = torch.matmul(qk_softmaxed, value)
# permute & reshape for further layer operations
fin = fin.permute([0, 2, 1, 3])
fin = fin.reshape([1, input_tensor.size()[1], 128])

print("dot_product: ", fin)

# Layer 1 - Self Output
weight_output_dense = model.bert.encoder.layer[1].attention.output.dense.weight.clone().detach().double().transpose(0, 1)
bias_output_dense = model.bert.encoder.layer[1].attention.output.dense.bias.clone().detach().double()

fin2 = torch.matmul(fin, weight_output_dense) + bias_output_dense
fin2_backup = fin2.clone()
fin2_backup = fin2_backup + input_tensor

fin3_whole = []

for i in range(len(input_tensor.squeeze())):
    fin2 = fin2_backup.squeeze()[i]
    fin3_corr = (fin2.squeeze().detach() - mean_1_0[i]) * inv_sqrt_var_1_0[i]

    weight_output_layernorm = model.bert.encoder.layer[1].attention.output.LayerNorm.weight.clone().detach().double().unsqueeze(0)
    bias_output_layernorm = model.bert.encoder.layer[1].attention.output.LayerNorm.bias.clone().detach().double()

    fin3_corr = fin3_corr * weight_output_layernorm + bias_output_layernorm
    fin3_whole.append(fin3_corr.detach())

fin3_whole = torch.cat(tuple(fin3_whole), 0).unsqueeze(0)

# Layer 1 - Intermediate
fin_4 = torch.matmul(fin3_whole, model.bert.encoder.layer[1].intermediate.dense.weight.transpose(0, 1).double())
+ model.bert.encoder.layer[1].intermediate.dense.bias
fin_5 = torch.nn.functional.gelu(fin_4)

# Layer 1 - Output
fin_6 = torch.matmul(fin_5, model.bert.encoder.layer[1].output.dense.weight.transpose(0, 1).double())
+ model.bert.encoder.layer[1].output.dense.bias

fin7_whole = []
for i in range(len(input_tensor.squeeze())):
    fin_7 = fin_6.squeeze()[i]

    fin7_corr = (fin_7.squeeze().detach() - mean_1_1[i]) * inv_sqrt_var_1_1[i]

    weight_output_layernorm = model.bert.encoder.layer[1].attention.output.LayerNorm.weight.clone().detach().double().unsqueeze(0)
    bias_output_layernorm = model.bert.encoder.layer[1].attention.output.LayerNorm.bias.clone().detach().double()

    fin7_corr = fin7_corr * weight_output_layernorm + bias_output_layernorm
    fin7_whole.append(fin7_corr.detach())

fin7_whole = torch.cat(tuple(fin7_whole), 0).unsqueeze(0)

# Pooler
pooler_output = torch.tanh(torch.matmul(fin7_whole.double(), model.bert.pooler.dense.weight.transpose(0, 1).double())
                           + model.bert.pooler.dense.bias)

# Classifier
classification = torch.matmul(pooler_output, model.classifier.weight.transpose(0, 1).double()
                              + model.classifier.bias.double())

# Output
print("Plain circuit output: {}".format(classification[0][0].detach().numpy()))
