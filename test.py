import transformers
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, TrainingArguments, logging, \
    DataCollatorWithPadding, BartForSequenceClassification
from datasets import load_dataset, load_metric, get_dataset_split_names, Features, ClassLabel, Dataset
import torch
import os
import configparser
import numpy as np
import sys
import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from datetime import datetime

config = configparser.ConfigParser()
config.sections()
config.read('config.ini')
config_dict = {}
for var in config['VARIABLES']:
    value = config['VARIABLES'][var]
    if value.isnumeric():
        value = int(value)
    config_dict[var] = value

model = BartForSequenceClassification.from_pretrained('models/BART-pairwise/accuracy/test')
tokenizer = BartTokenizer.from_pretrained('models/BART-pairwise/accuracy/test')
