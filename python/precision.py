from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import math
from matplotlib import pyplot as plt 
from datasets import load_dataset
import pandas as pd

def precision(correct, approx):
    if type(approx) == list:
        approx = np.array(approx)
    absolute = sum(abs(correct - approx))/len(correct)
    relative = absolute / (sum(abs(correct))/len(correct))
    return 1 - relative

def relative_error(correct, approx):
    relative_errors = abs(correct - approx) / max(correct)
    return sum(relative_errors)/len(relative_errors)

from transformers import logging
logging.set_verbosity_error() #Otherwise it will log annoying warnings

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
trained = torch.load('SST-2-BERT-tiny.bin', map_location=torch.device('cpu'))
model.load_state_dict(trained , strict=True)

model.eval()

text = "Nuovo Cinema Paradiso has been an incredible movie! A gem in the italian culture."
text = "[CLS] " + text + " [SEP]"

#This is computed client-side

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

fhe_vector = np.array([ -0.3090, -0.0246,  0.7970, -0.0238,  0.1896,  0.3124,  0.0414, -0.2285, -0.7296, -0.3780,  0.2053, -0.3971, -0.3614,  0.0559, -0.6637, -0.3618, -0.8222, -0.0580,  0.6474,  0.1623, -0.2207, -0.1006, -0.1696, -0.0141, -0.2170,  0.2289,  0.3672, -0.2401, -0.2847, -0.4943, -0.1021,  0.3427,  0.2066, -0.1300,  0.1291,  0.8506, -0.6453, -0.6731, -0.1210,  0.3211,  0.0155, -0.2310,  0.6582,  0.1582,  0.1238, -0.3713,  0.5834,  0.1905, -0.1636,  0.3664, -0.2616,  0.0522,  0.5595,  0.2635, -0.7683,  0.2608,  0.5117,  0.5679,  0.0526,  0.6444,  0.5096, -0.7960,  0.0409,  0.3002,  0.0493,  0.1228,  0.8845, -0.5277,  0.8978,  0.0986,  0.1151,  0.5468,  1.2089, -0.1721,  0.8912,  0.3563,  0.5092,  0.2152,  0.1775, -0.2963, -0.5777, -0.4493,  0.2931, -0.2802,  0.6357, -0.2518, -1.1975, -0.3656, -0.6256, -0.6295,  0.2502, -0.0918,  0.9606,  0.2442,  0.8670,  0.3603,  0.6010, -0.4238,  0.2473,  0.8414,  0.8304,  0.7751,  0.2766,  0.2580, -0.7249,  0.5046, -0.3302,  0.5718, -0.5278,  1.1236,  1.7388, -0.3552, -0.3844,  1.1474, -0.4412,  0.6235, -0.5075, -0.6485, -0.1250,  0.2654,  1.2387,  0.7896, -0.5817, -0.1745, -0.0075, -0.2276,  1.3081, -0.9385 ])

precision(fin[0][0].detach(), fhe_vector)