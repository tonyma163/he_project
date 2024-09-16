from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import math
from matplotlib import pyplot as plt 
from datasets import load_dataset
import pandas as pd

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

# Load fine-tuned model weights
trained = torch.load('SST-2-BERT-tiny.bin', map_location=torch.device('cpu'))
# Remove the unexpected key if present
trained.pop('bert.embeddings.position_ids', None)
model.load_state_dict(trained, strict=True)

# Set model to evaluation mode
model.eval()

# Load SST-2 validation dataset
dataset = pd.read_parquet("SST-2-val.parquet")

#type = "./precomputed"
#type = "./precomputed2"
type = "./precomputed3"

# ./precomputed
"""
10/10 of the dataset
Accuracy plain: 0.8371559633027523
Accuracy precom: 0.8130733944954128
Performance loss: 0.02876712328767128
"""

# ./precomputed2
"""
1/10 of the dataset
Accuracy plain: 0.8371559633027523
Accuracy precom: 0.8153669724770642
Performance loss: 0.026027397260274032
"""

# ./precomputed3
"""
1/100 of the dataset
Accuracy plain: 0.8371559633027523
Accuracy precom: 0.8142201834862385
Performance loss: 0.027397260273972712
"""

# Load precomputed means and variances
mean_0_0 = np.loadtxt(f"{type}/layer0_self_output_mean.txt")
inv_sqrt_var_0_0 = np.loadtxt(f"{type}/layer0_self_output_inv_sqrt_var.txt")

mean_0_1 = np.loadtxt(f"{type}/layer0_output_mean.txt")
inv_sqrt_var_0_1 = np.loadtxt(f"{type}/layer0_output_inv_sqrt_var.txt")

mean_1_0 = np.loadtxt(f"{type}/layer1_self_output_mean.txt")
inv_sqrt_var_1_0 = np.loadtxt(f"{type}/layer1_self_output_inv_sqrt_var.txt")

mean_1_1 = np.loadtxt(f"{type}/layer1_output_mean.txt")
inv_sqrt_var_1_1 = np.loadtxt(f"{type}/layer1_output_inv_sqrt_var.txt")

print("mean_0_0: ", mean_0_0)

correct_plain = 0
correct_precomp = 0

for ind in dataset.index:
    text = "[CLS] " + dataset['sentence'][ind] + " [SEP]"

    tokenized = tokenizer(text)
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
        
    x = model.bert.embeddings(tokens_tensor, torch.tensor([[1] * len(tokenized_text)]))

    key = model.bert.encoder.layer[0].attention.self.key.weight.clone().detach().double().transpose(0, 1)
    query = model.bert.encoder.layer[0].attention.self.query.weight.clone().detach().double().transpose(0, 1)
    value = model.bert.encoder.layer[0].attention.self.value.weight.clone().detach().double().transpose(0, 1)

    key_bias = model.bert.encoder.layer[0].attention.self.key.bias.clone().detach().double()
    query_bias = model.bert.encoder.layer[0].attention.self.query.bias.clone().detach().double()
    value_bias = model.bert.encoder.layer[0].attention.self.value.bias.clone().detach().double()

    original_input_tensor = x.double()

    input_tensor = x.double()

    q = torch.matmul(input_tensor, query) + query_bias
    k = torch.matmul(input_tensor, key) + key_bias
    v = torch.matmul(input_tensor, value) + value_bias

    q = q.reshape([1, input_tensor.size()[1], 2, 64])
    k = k.reshape([1, input_tensor.size()[1], 2, 64])
    v = v.reshape([1, input_tensor.size()[1], 2, 64])

    q = q.permute([0, 2, 1, 3])
    k = k.permute([0, 2, 3, 1])

    qk = torch.matmul(q, k)
    qk = qk / 8

    qk_softmaxed = torch.softmax(qk, -1)

    v = v.permute([0, 2, 1, 3])

    fin = torch.matmul(qk_softmaxed, v)
    fin = fin.permute([0, 2, 1, 3])
    fin = fin.reshape([1, input_tensor.size()[1], 128])
    
    mean = mean_0_0
    var = inv_sqrt_var_0_0
    
    w_output_dense = model.bert.encoder.layer[0].attention.output.dense.weight.clone().detach().double().transpose(0, 1)
    b_output_dense = model.bert.encoder.layer[0].attention.output.dense.bias.clone().detach().double()

    fin2 = torch.matmul(fin, w_output_dense) + b_output_dense
    fin2_backup = fin2.clone()
    fin2_backup = fin2_backup + original_input_tensor

    fin3_whole = []

    for i in range(len(original_input_tensor.squeeze())):
        fin2 = fin2_backup.squeeze()[i]
        
        idx = i
        
        if i > len(mean) - 1:
            idx = len(mean) - 1
            
        fin3_corr = (fin2.squeeze().detach() - mean[idx]) * var[idx]

        w_output_layernorm = model.bert.encoder.layer[0].attention.output.LayerNorm.weight.clone().detach().double().unsqueeze(0)
        b_output_layernorm = model.bert.encoder.layer[0].attention.output.LayerNorm.bias.clone().detach().double()

        fin3_corr = fin3_corr * w_output_layernorm + b_output_layernorm
        fin3_whole.append(fin3_corr.detach())

    fin3_whole = torch.cat(tuple(fin3_whole), 0).unsqueeze(0)
    fin_4 = torch.matmul(fin3_whole, model.bert.encoder.layer[0].intermediate.dense.weight.transpose(0, 1).double()) + model.bert.encoder.layer[0].intermediate.dense.bias
    
    fin_5 = torch.nn.functional.gelu(fin_4)
    fin_6 = torch.matmul(fin_5, model.bert.encoder.layer[0].output.dense.weight.transpose(0, 1).double()) + model.bert.encoder.layer[0].output.dense.bias
    fin_6 = fin_6 + fin3_whole
    
    mean = mean_0_1
    var = inv_sqrt_var_0_1
    
    fin7_whole = []

    for i in range(len(input_tensor.squeeze())):
        fin_7 = fin_6.squeeze()[i]
        
        idx = i
        
        if i > len(mean) - 1:
            idx = len(mean) - 1
            
        fin7_corr = (fin_7.squeeze().detach() - mean[idx]) * var[idx]

        w_output_layernorm = model.bert.encoder.layer[0].output.LayerNorm.weight.clone().detach().double().unsqueeze(0)
        b_output_layernorm = model.bert.encoder.layer[0].output.LayerNorm.bias.clone().detach().double()

        fin7_corr = fin7_corr * w_output_layernorm + b_output_layernorm

        fin7_whole.append(fin7_corr.detach())

    fin7_whole = torch.cat(tuple(fin7_whole), 0).unsqueeze(0)
    
    key = model.bert.encoder.layer[1].attention.self.key.weight.clone().detach().double().transpose(0, 1)
    query = model.bert.encoder.layer[1].attention.self.query.weight.clone().detach().double().transpose(0, 1)
    value = model.bert.encoder.layer[1].attention.self.value.weight.clone().detach().double().transpose(0, 1)

    key_bias = model.bert.encoder.layer[1].attention.self.key.bias.clone().detach().double()
    query_bias = model.bert.encoder.layer[1].attention.self.query.bias.clone().detach().double()
    value_bias = model.bert.encoder.layer[1].attention.self.value.bias.clone().detach().double()

    original_input_tensor = fin7_whole
    input_tensor = fin7_whole

    q = torch.matmul(input_tensor, query) + query_bias
    k = torch.matmul(input_tensor, key) + key_bias
    v = torch.matmul(input_tensor, value) + value_bias

    q = q.reshape([1, input_tensor.size()[1], 2, 64])
    k = k.reshape([1, input_tensor.size()[1], 2, 64])
    v = v.reshape([1, input_tensor.size()[1], 2, 64])

    q = q.permute([0, 2, 1, 3])
    k = k.permute([0, 2, 3, 1])

    qk = torch.matmul(q, k)
    qk = qk / 8

    qk_softmaxed = torch.softmax(qk, -1)

    v = v.permute([0, 2, 1, 3])

    fin = torch.matmul(qk_softmaxed, v)
    fin = fin.permute([0, 2, 1, 3])
    fin = fin.reshape([1, input_tensor.size()[1], 128])
    
    mean = mean_1_0
    var = inv_sqrt_var_1_0
    
    w_output_dense = model.bert.encoder.layer[1].attention.output.dense.weight.clone().detach().double().transpose(0, 1)
    b_output_dense = model.bert.encoder.layer[1].attention.output.dense.bias.clone().detach().double()

    fin2 = torch.matmul(fin, w_output_dense) + b_output_dense
    fin2_backup = fin2.clone()
    fin2_backup = fin2_backup + original_input_tensor

    fin3_whole = []

    for i in range(len(original_input_tensor.squeeze())):
        fin2 = fin2_backup.squeeze()[i]

        idx = i
        
        if i > len(mean) - 1:
            idx = len(mean) - 1
            
        fin3_corr = (fin2.squeeze().detach() - mean[idx]) * var[idx]

        w_output_layernorm = model.bert.encoder.layer[1].attention.output.LayerNorm.weight.clone().detach().double().unsqueeze(0)
        b_output_layernorm = model.bert.encoder.layer[1].attention.output.LayerNorm.bias.clone().detach().double()

        fin3_corr = fin3_corr * w_output_layernorm + b_output_layernorm
        fin3_whole.append(fin3_corr.detach())

    fin3_whole = torch.cat(tuple(fin3_whole), 0).unsqueeze(0)
    fin_4 = torch.matmul(fin3_whole, model.bert.encoder.layer[1].intermediate.dense.weight.transpose(0, 1).double()) + model.bert.encoder.layer[1].intermediate.dense.bias
    
    fin_5 = torch.nn.functional.gelu(fin_4)
    
    fin_6 = torch.matmul(fin_5, model.bert.encoder.layer[1].output.dense.weight.transpose(0, 1).double()) + model.bert.encoder.layer[1].output.dense.bias
    fin_6 = fin_6 + fin3_whole
    
    fin7_whole = []
    
    mean = mean_1_1
    var = inv_sqrt_var_1_1

    for i in range(len(input_tensor.squeeze())):
        fin_7 = fin_6.squeeze()[i]

        idx = i
        
        if i > len(mean) - 1:
            idx = len(mean) - 1
            
        fin7_corr = (fin_7.squeeze().detach() - mean[idx]) * var[idx]

        w_output_layernorm = model.bert.encoder.layer[1].output.LayerNorm.weight.clone().detach().double().unsqueeze(0)
        b_output_layernorm = model.bert.encoder.layer[1].output.LayerNorm.bias.clone().detach().double()

        fin7_corr = fin7_corr * w_output_layernorm + b_output_layernorm

        fin7_whole.append(fin7_corr.detach())

    fin7_whole = torch.cat(tuple(fin7_whole), 0).unsqueeze(0)

    densed_pooler = torch.tanh(torch.matmul(fin7_whole.double(), model.bert.pooler.dense.weight.transpose(0, 1).double()) + model.bert.pooler.dense.bias)

    approx = densed_pooler[0][0].detach()
    precise = model.bert.pooler(model.bert.encoder(x)[0]).detach()[0]
    
    output = torch.matmul(approx, model.classifier.weight.transpose(0, 1).double()) + model.classifier.bias.double()
    output_real = model(tokens_tensor, torch.tensor([[1] * len(tokenized_text)])).logits[0].detach()

    
    if output_real[0].item() > output_real[1].item() and dataset['label'][ind] == 0:
        correct_plain = correct_plain + 1
    if output_real[0].item() < output_real[1].item() and dataset['label'][ind] == 1:
        correct_plain = correct_plain + 1
        
    if output[0].item() > output[1].item() and dataset['label'][ind] == 0:
        correct_precomp = correct_precomp + 1
    if output[0].item() < output[1].item() and dataset['label'][ind] == 1:
        correct_precomp = correct_precomp + 1
        
accuracy_plain = float(correct_plain) / len(dataset['label'])
accuracy_precomp = float(correct_precomp) / len(dataset['label'])

print("Accuracy plain: {}\nAccuracy precom: {}".format(accuracy_plain, accuracy_precomp))

print("Performance loss: {}".format(1 - accuracy_precomp / accuracy_plain))