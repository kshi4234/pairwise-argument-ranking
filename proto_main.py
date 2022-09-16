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

separator = '[SEP]'

# This line below enforces visible GPUS (4 in total for my remote machine)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
torch_device = device

loaded_metrics = None


# Self defined metric for use in the HuggingFace Trainer
def compute_metrics(e):
    preds = e.predictions[0] if isinstance(e.predictions, tuple) else e.predictions
    preds = np.argmax(preds, axis=1)
    result = loaded_metrics.compute(predictions=preds, references=e.label_ids)
    # print(preds)
    # print(e.label_ids)
    if len(result) > 1:
        result['combined_score'] = np.mean(list(result.values())).item()
    return result


# Set up the process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


# K-Fold Cross Validation
class KTrainer:
    def __init__(self):
        self.config_dict = {}
        self.total_data = ''
        self.data_collator = ''
        self.labels = ClassLabel(num_classes=2, names=['a1', 'a2'])
        self.tokenizer = ''
        self.model = ''
        self.data_collator = ''

    # Load the model from the configuration
    def load_model(self, model_name):
        self.tokenizer = BartTokenizer.from_pretrained(model_name, num_labels=2)
        self.model = BartForSequenceClassification.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    # Load the configuration file with necessary contents
    def load_config(self, ini_file):
        config = configparser.ConfigParser()
        config.sections()
        config.read(ini_file)
        for var in config['VARIABLES']:
            value = config['VARIABLES'][var]
            if value.isnumeric():
                value = int(value)
            self.config_dict[var] = value

    # Load the total set and do initial splitting
    def load_dataset(self):
        self.total_data = pd.read_csv(self.config_dict['train_file'])
        print(self.total_data.columns)
        # print(self.total_data)

    # Begin cross-validation
    def train_help(self, model_path, topic, do_train, do_test):
        # suppose we have 4 gpus
        m = PairwiseModel(self.config_dict, self.labels, self.model,
                          self.data_collator, self.tokenizer)
        m.preprocess(self.total_data, topic)
        # m.load_trainer(m.train_set, m.test_set)
        mp.spawn(m.load_trainer, args=(4, m.train_set, m.test_set, model_path, do_train, do_test), nprocs=4, join=True)

    def train(self, model_path):
        topics = self.total_data.topic.unique()
        for topic in topics:
            save_path = os.path.join(model_path, topic)
            if not os.path.isdir(save_path):
                self.train_help(save_path, topic, True, False)
            else:
                done_model = BartForSequenceClassification.from_pretrained(save_path)
                done_tokenizer = BartTokenizer.from_pretrained(save_path)
                self.train_help(save_path, topic, False, True)
                print('SKIPPED ' + save_path)
            torch.cuda.empty_cache()
        return


class PairwiseModel:
    # The constructor for the class
    def __init__(self, config_dict, labels, model, data_collator, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.config_dict = config_dict
        self.trainer = ''
        self.data_collator = data_collator
        self.labels = labels
        self.train_set = ''
        self.test_set = ''

    # The destructor
    def __del__(self):
        print("MODEL IS DONE. NOW MODEL IS DEAD!")

    # Slice the training set to make it a specified size
    def subset_train(self):
        self.train_set = self.train_set.shuffle().select(
            range(int(float(self.config_dict['set_ratio']) * len(self.train_set))))
        return

    # Slice the training set to make it a specified size
    def subset_test(self):
        self.test_set = self.test_set.shuffle().select(
            range(int(float(self.config_dict['set_ratio']) * len(self.test_set))))
        return

    # Tokenize the input specified in config
    def tokenize_input(self, data):
        return self.tokenizer(data[self.config_dict['input']], padding='max_length', truncation=True)

    # Tokenize the output specified in config
    def relabel(self, data):
        data[self.config_dict['output']] = self.tokenizer.encode(data[self.config_dict['output']], padding='max_length',
                                                                 truncation=True)
        return data

    # Change all labels to integer values
    def to_int(self, data):
        ans = []
        for label in data['label'][1]:
            if label == '1':
                ans.append(0)
            else:
                ans.append(1)
        data['label'] = ans
        return data

    def preprocess(self, total_data, topic):
        # Train set is everything except for selected topic
        self.train_set = Dataset.from_pandas(total_data[total_data.topic != topic])
        # Test set is the topic left out (leave one out)
        self.test_set = Dataset.from_pandas(total_data[total_data.topic == topic])
        # Make the dataset smaller
        self.subset_train()
        self.subset_test()
        # -- DEBUGGING
        # print(self.train_set[0])
        # print(self.test_set[0])
        # --
        # Tokenize training set and relabel the outputs
        self.train_set = self.train_set.map(self.tokenize_input)  # , batched=True)
        self.train_set = self.train_set.rename_column(self.config_dict['output'], 'label')
        self.train_set = self.train_set.map(self.to_int)
        # Tokenize evaluation set and relabel the outputs
        self.test_set = self.test_set.map(self.tokenize_input)  # , batched=True)
        self.test_set = self.test_set.rename_column(self.config_dict['output'], 'label')
        self.test_set = self.test_set.map(self.to_int)

    # Function to create the trainer
    def load_trainer(self, rank, world_size, tokenized_train, tokenized_test, save_path, do_train, do_test):
        setup(rank, world_size)
        global loaded_metrics
        loaded_metrics = load_metric(self.config_dict['metric'])
        training_args = TrainingArguments(output_dir="../test_trainer",
                                          num_train_epochs=self.config_dict['num_epochs'],
                                          logging_strategy='steps',
                                          per_device_train_batch_size=int(self.config_dict['batch_size']),
                                          per_device_eval_batch_size=int(self.config_dict['batch_size']),
                                          gradient_accumulation_steps=int(self.config_dict['gradient_steps']),
                                          save_steps=int(self.config_dict['save_steps']),
                                          local_rank=rank)

        self.trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            tokenizer=self.tokenizer,
            eval_dataset=tokenized_test,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )
        print("build trainer with on device:", training_args.device, "with n gpus:", training_args.n_gpu)
        start = datetime.now()
        # Begin training
        if do_train:
            self.trainer.train()
            if rank == 0:
                # Save fine-tuned model
                self.model.save_pretrained(save_path)
                # Save tokenizer for fine-tuned model
                self.tokenizer.save_pretrained(save_path)
        if do_test:
            print(self.gen_predict(self.test_set).metrics)
        print(f"finished in {datetime.now() - start} seconds")

    def gen_predict(self, test):
        print('TEST: ', test)
        predictions = self.trainer.predict(test, ignore_keys=['title', 'stance', 'args', 'reason'])
        return predictions


# STEPS:
# 1. Create KTrainer
# 2. Load configuration file
# 3. Load total dataset
# 4. Load model
# 5. Begin Training
def main():
    # Create instance of KTrainer
    k = KTrainer()
    # Load the configuration file
    k.load_config('config.ini')
    # Path to saved model
    model_path = k.config_dict['model_path']
    # Load the dataset specified in config.ini
    k.load_dataset()
    # Load base model
    k.load_model(k.config_dict['model_name'])
    # Begin training with k-folds
    k.train(os.path.join(model_path, k.config_dict['metric']))


if __name__ == '__main__':
    main()
