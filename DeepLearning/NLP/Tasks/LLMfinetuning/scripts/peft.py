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

from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig

from tqdm import tqdm
tqdm.pandas()
import os

# Зададим расположения кэша моделей и укажем GPU, на котором будет исполняться ноутбук.
# Эти 3 строки нужно модифицировать в соответствии с конфигурацией среды, в которой запускается код.
os.environ['HF_HOME'] = '/home/jovyan/work/ramdisk/shlyahin/HF_cache/'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

huggingface_dataset_name = 'IlyaGusev/gazeta' #"UrukHan/t5-russian-spell_I"
dataset = load_dataset(huggingface_dataset_name)

model_max_length = 512
model_name='cointegrated/rut5-base-multitask'
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


lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

peft_model = get_peft_model(original_model, lora_config)

output_dir = f'../reports/peft-2-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-4, 
    num_train_epochs=10,
    logging_steps=1000,
    save_steps=5000,
    evaluation_strategy ='steps',
    eval_steps=1000,
    load_best_model_at_end=True,
    report_to="wandb"
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets['validation']
)

peft_trainer.train()

peft_model_path = "../weights/peft_2"
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)