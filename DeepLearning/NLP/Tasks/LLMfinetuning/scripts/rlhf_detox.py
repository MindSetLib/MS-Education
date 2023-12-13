#!/usr/bin/env python
# coding: utf-8
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
import torch
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, GenerationConfig, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from tqdm import tqdm
tqdm.pandas()
import os

from peft import PeftModel, PeftConfig, LoraConfig, TaskType

# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler


import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level='ERROR')

# Зададим расположения кэша моделей и укажем GPU, на котором будет исполняться ноутбук.
# Эти 3 строки нужно модифицировать в соответствии с конфигурацией среды, в которой запускается код.
os.environ['HF_HOME'] = '/home/jovyan/work/ramdisk/shlyahin/HF_cache/'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4" #4

os.environ["TOKENIZERS_PARALLELISM"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

device = torch.device('cuda')

model_name = 'BlackSamorez/rudialogpt3_medium_based_on_gpt2_2ch'
original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

tokenizer.pad_token = tokenizer.eos_token

collator = DataCollatorWithPadding(tokenizer, max_length=512, padding=True)

dataset = load_from_disk("../data/dialogues3.hf")
print(dataset)

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(original_model,
                                                               torch_dtype=torch.bfloat16,
                                                               is_trainable=True).to(device)

ref_model = create_reference_model(ppo_model)

toxicity_model_name = "sismetanin/rubert-toxic-pikabu-2ch"
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name).to(device)

# get the logits for "not hate" - this is the reward!
not_hate_index = 0
tox_label = 'LABEL_1'
non_tox_label = 'LABEL_0'

device_ = 0 if device == 'cuda' else "cpu"

sentiment_pipe = pipeline("sentiment-analysis", 
                          model=toxicity_model_name, 
                          device=device_)
reward_logits_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none", # Set to "none" to retrieve raw logits.
    "batch_size": 16
}

reward_probabilities_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "softmax", # Set to "softmax" to apply softmax and retrieve probabilities.
    "batch_size": 16
}

toxicity_evaluator = evaluate.load("toxicity", 
                                    toxicity_model_name,
                                    module_type="measurement",
                                    toxic_label="toxic")

max_new_tokens = 64

def evaluate_toxicity(model, 
                      toxicity_evaluator, 
                      tokenizer, 
                      dataset, 
                      num_samples):
    
    toxicities = []
    input_texts = []
    for i, sample in tqdm(enumerate(dataset)):
        input_text = sample["query"]

        if i > num_samples:
            break
            
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids.to(device)
        
        generation_kwargs = {'bad_words_ids': [[tokenizer.pad_token_id]],
                             'temperature': 1.,
                             #'num_beams': 3,
                             'repetition_penalty': 10.,
                             'do_sample': True}
        
        generation_kwargs['max_length'] = len(input_ids[0]) + max_new_tokens

        response_token_ids = model.generate(input_ids=input_ids,
                                            **generation_kwargs)
        
        generated_text = tokenizer.decode(response_token_ids[:, -max_new_tokens:][0], skip_special_tokens=True)
        
        toxicity_score = toxicity_evaluator.compute(predictions=[(input_text + " " + generated_text)], toxic_label=tox_label)

        toxicities.extend(toxicity_score["toxicity"])

    # Compute mean & std using np.
    mean = np.mean(toxicities)
    std = np.std(toxicities)
        
    return mean, std


mean_before_detoxification, std_before_detoxification = evaluate_toxicity(model=ref_model, 
                                                                          toxicity_evaluator=toxicity_evaluator, 
                                                                          tokenizer=tokenizer, 
                                                                          dataset=dataset["test"], 
                                                                          num_samples=100)

def get_reward(reward):
    for item in reward:
        if item['label'] == non_tox_label:
            return item['score']
    return None

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


learning_rate =  2 * 1.47e-5 #1.41e-5
max_ppo_epochs = 100 #4
mini_batch_size = 4 #4
batch_size = 16 #16

config = PPOConfig(
    model_name=model_name,    
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size
)

ppo_trainer = PPOTrainer(config=config, 
                         model=ppo_model, 
                         ref_model=ref_model, 
                         tokenizer=tokenizer, 
                         dataset=dataset["train"], 
                         data_collator=collator)

generation_kwargs = {'bad_words_ids': [[tokenizer.pad_token_id]],
                     'temperature': 1.,
                     #'num_beams': 3,
                     'repetition_penalty': 10.,
                     'do_sample': True}

reward_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none", # You want the raw logits without softmax.
    "batch_size": 16
}

max_ppo_steps = 25 #50
metrics = []

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # Break when you reach max_steps.
    if step >= max_ppo_steps:
        break   

    prompt_tensors = batch["input_ids"]

    # Get response from LLM.
    response_tensors = []

    for prompt_tensor in prompt_tensors:

        generation_kwargs['max_length'] = len(prompt_tensor) + max_new_tokens

        response = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
        
        response_tensors.append(response.squeeze()[-max_new_tokens :])
        
    # This needs to be called "response".
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute reward outputs.
    query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])] 
    rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)

    # You use the `nothate` item because this is the score for the positive `nothate` class.
    reward_tensors = [torch.tensor(get_reward(reward)) for reward in rewards]    

    # Run PPO step.
    stats = ppo_trainer.step(prompt_tensors, response_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)
    
    metrics.append(stats)
    print(f'objective/kl: {stats["objective/kl"]}')
    print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
    print('-'.join('' for x in range(100)))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('../reports/rlhf_100ppoe.csv')
    

mean_after_detoxification, std_after_detoxification = evaluate_toxicity(model=ppo_model, 
                                                                        toxicity_evaluator=toxicity_evaluator, 
                                                                        tokenizer=tokenizer, 
                                                                        dataset=dataset["test"], 
                                                                        num_samples=100)
print(f'toxicity [mean, std] after detox: [{mean_after_detoxification}, {std_after_detoxification}]')


mean_improvement = (mean_before_detoxification - mean_after_detoxification) / mean_before_detoxification
std_improvement = (std_before_detoxification - std_after_detoxification) / std_before_detoxification

print(f'Percentage improvement of toxicity score after detoxification:')
print(f'mean: {mean_improvement*100:.2f}%')
print(f'std: {std_improvement*100:.2f}%')

ppo_trainer.save_pretrained(f'../weights/ppo_tox/dvach-new3-{max_ppo_steps}')



