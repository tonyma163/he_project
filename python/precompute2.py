import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# Load tokenizer & model
logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny-mnli")
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

# Load fine-tuned model weights
trained_weight = torch.load('SST-2-BERT-tiny.bin', map_location=torch.device('cpu'))
trained_weight.pop('bert.embeddings.position_ids', None) # remove unexpected keys
model.load_state_dict(trained_weight, strict=True) # load fine-tuned weight into model 
#print(model)

# Testing
model.eval()

# Load training dataset
dataset = load_dataset("stanfordnlp/sst2", split="train")

# if only 1/10 dataset
num_samples = int(0.1 * len(dataset))
dataset = dataset.shuffle(seed=42)
subset_dataset = dataset.select(range(num_samples))

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example['sentence'], truncation=True, padding='max_length', max_length=128)
tokenized_dataset = subset_dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create a dataloader
dataloader = DataLoader(tokenized_dataset, batch_size=32)

# Create a dictionary to store LayerNorm Inputs
layernorm_inputs = {
    'layer0_self_output': [],
    'layer0_output': [],
    'layer1_self_output': [],
    'layer1_output': []
}

# Hook function to capture the inputs from each LayerNorm layer
def get_layernorm_input(layer):
    def hook(module, input):
        layernorm_inputs[layer].append(input[0].detach())
    return hook
layer0_self_output_hook = model.bert.encoder.layer[0].attention.output.LayerNorm.register_forward_pre_hook(
    get_layernorm_input('layer0_self_output')
)
layer0_output_hook = model.bert.encoder.layer[0].output.LayerNorm.register_forward_pre_hook(
    get_layernorm_input('layer0_output')
)
layer1_self_output_hook = model.bert.encoder.layer[1].attention.output.LayerNorm.register_forward_pre_hook(
    get_layernorm_input('layer1_self_output')
)
layer1_output_hook = model.bert.encoder.layer[1].output.LayerNorm.register_forward_pre_hook(
    get_layernorm_input('layer1_output')
)

# Process
attention_mask_list = [] # to excludle padding tokens
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        attention_mask_list.append(attention_mask.detach())
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Clean the hooks
layer0_self_output_hook.remove()
layer0_output_hook.remove()
layer1_self_output_hook.remove()
layer1_output_hook.remove()

# to excludle padding tokens
all_attention_masks = torch.cat(attention_mask_list, dim=0)
all_attention_masks = all_attention_masks.view(-1)

# Compute the mean & inverse sqrt variance for each LayerNorm layer
for layer, input_list in layernorm_inputs.items():
    # concatenate all inputs for the current layer
    all_inputs = torch.cat(input_list, dim=0)

    # flatten the inputs to merge batch and sequence dimensions
    total_samples, seq_length, hidden_size = all_inputs.shape
    all_inputs = all_inputs.view(-1, hidden_size)
    
    # exclude padding tokens
    valid_indices = all_attention_masks.nonzero(as_tuple=False).squeeze()
    valid_inputs = all_inputs[valid_indices]

    # Compute mean and variance across all tokens and samples for each feature
    mean = valid_inputs.mean(dim=0)
    var = valid_inputs.var(dim=0, unbiased=False)

    # Compute the inverse square root of variance + epsilon
    epsilon = 1e-12
    inv_sqrt_var = 1.0 / torch.sqrt(var + epsilon)

    #
    path = "./precomputed2"
    if not (os.path.exists(path)):
        os.makedirs(path)
    # Save the means & inverse sqrt variance to text files
    np.savetxt(f"./{path}/{layer}_mean.txt", mean.numpy())
    np.savetxt(f"./{path}/{layer}_inv_sqrt_var.txt", inv_sqrt_var.numpy())

print("completed.")