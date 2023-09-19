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
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device('cuda')
huggingface_dataset_name = 'IlyaGusev/gazeta' 
dataset = load_dataset(huggingface_dataset_name)

model_name = 'cointegrated/rut5-base-multitask'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

instruct_model_path = '../weights/ift_2'
instruct_model = AutoModelForSeq2SeqLM.from_pretrained(instruct_model_path, torch_dtype=torch.bfloat16).to(device)

from peft import PeftModel, PeftConfig
peft_model_path = '../weights/peft_2'
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
peft_model = PeftModel.from_pretrained(peft_model_base, 
                                       peft_model_path, 
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False).to(device)

fract = 1000

texts = dataset['test'][0:fract]['text']
human_baseline_summaries = dataset['test'][0:fract]['summary']

original_model_summaries = []
instruct_model_summaries = []
peft_model_summaries = []

for idx, text in tqdm(enumerate(texts)):
    
    prompt = f'Summarize | {text}' 
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    human_baseline_text_output = human_baseline_summaries[idx]
    
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0][2:], skip_special_tokens=True)

    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0][2:], skip_special_tokens=True)

    original_model_summaries.append(original_model_text_output)
    instruct_model_summaries.append(instruct_model_text_output)
    peft_model_summaries.append(peft_model_text_output)

zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries, peft_model_summaries))
 
df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'instruct_model_summaries', 'peft_model_summaries'])

df.to_csv('../reports/test2.csv')