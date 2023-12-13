from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

from tqdm import tqdm
tqdm.pandas()
import os

# Зададим расположения кэша моделей и укажем GPU, на котором будет исполняться ноутбук.
# Эти 3 строки нужно модифицировать в соответствии с конфигурацией среды, в которой запускается код.
os.environ['HF_HOME'] = '/home/jovyan/work/ramdisk/shlyahin/HF_cache/'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

huggingface_dataset_name = 'IlyaGusev/gazeta' 
dataset = load_dataset(huggingface_dataset_name)



model_max_length = 512

t_config = {"learning_rate": 0.0004,
             "num_train_epochs": 8,
             "gradient_accumulation_steps": 32,
             "logging_steps": 25,
             "save_steps": 250,
             "eval_steps": 250,
             "warmup_steps": 100,
             "weight_decay": 0,
            }


models = ['cointegrated/rut5-base-multitask']

for i, model_name in enumerate(models):
    

    
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=model_max_length)

    def tokenize_function(example):

        prompt = ['Summarize | ' + ' '.join(text.split(' ')[:model_max_length - 10]) for text in example["text"]]
        example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
        example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

        return example

    # The dataset actually contains 3 diff splits: train, validation, test.
    # The tokenize_function code is handling all data across all splits in batches.
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['title', 'date', 'url', 'text', 'summary',])


    output_dir = f'../reports/ift-2-training-{str(int(time.time()))}'

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=t_config["learning_rate"], # 1e-5,
        num_train_epochs=t_config["num_train_epochs"],
        gradient_accumulation_steps=t_config["gradient_accumulation_steps"],
        weight_decay=t_config["weight_decay"],
        logging_steps=t_config["logging_steps"],
        save_steps=t_config["save_steps"],
        evaluation_strategy ='steps',
        eval_steps=t_config["eval_steps"],
        warmup_steps=t_config["warmup_steps"],
        report_to="wandb",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=original_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation']
    )

    trainer.train()
    
    ft_model_path = f'../weights/ift_2'
    trainer.model.save_pretrained(ft_model_path)
    tokenizer.save_pretrained(ft_model_path)