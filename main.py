import transformers
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, TrainingArguments, logging, \
    DataCollatorWithPadding, BartForSequenceClassification
from datasets import load_dataset, load_metric, get_dataset_split_names, Features, ClassLabel
import torch
import os
import configparser
import numpy as np

separator = '[SEP]'

# This line below enforces single-GPU usage (don't know how to parallelize currently)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else torch.device("cpu")
torch_device = device

loaded_metrics = None


def compute_metrics(e):
    # print("INSTANCE: ", isinstance(e.predictions, tuple))
    preds = e.predictions[0] if isinstance(e.predictions, tuple) else e.predictions
    preds = np.argmax(preds, axis=1)
    result = loaded_metrics.compute(predictions=preds, references=e.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


class PairwiseModel:
    # The constructor for the class
    def __init__(self):
        self.tokenizer = ''
        self.model = ''
        self.config_dict = {}
        self.total_data = ''
        self.train_set = ''
        self.test_set = ''
        self.trainer = ''
        self.data_collator = ''
        self.labels = ClassLabel(num_classes=2, names=['a1', 'a2'])

    # Load the model from the configuration
    def load_model(self, model_name):
        self.tokenizer = BartTokenizer.from_pretrained(model_name, num_labels=2)
        self.model = BartForSequenceClassification.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    # Load the configuration file
    def load_config(self, ini_file):
        config = configparser.ConfigParser()
        config.sections()
        config.read(ini_file)
        for var in config['VARIABLES']:
            value = config['VARIABLES'][var]
            if value.isnumeric():
                value = int(value)
            self.config_dict[var] = value

    # Load the dataset from the path specified in config
    def load_dataset(self):
        if self.config_dict['test_file'] == 'None':
            self.total_data = load_dataset('csv', data_files=self.config_dict['train_file'], split='train')
            self.total_data = self.total_data.train_test_split(test_size=0.1)
            self.train_set, self.test_set = self.total_data['train'], self.total_data['test']
        else:
            self.train_set = load_dataset('csv', data_files=self.config_dict['train_file'], split='train')
            self.test_set = load_dataset('csv', data_files=self.config_dict['test_file'], split='train')

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

    def to_int(self, data):
        ans = []
        for label in data['label'][1]:
            if label == '1':
                # print(label, 0)
                ans.append(0)
            else:
                # print(label, 1)
                ans.append(1)
        data['label'] = ans
        return data

    def map_to_labels(self, data):
        data['label'] = self.labels.int2str(data['label'])
        return data

    # Function to create the trainer
    def load_trainer(self, tokenized_train, tokenized_test):
        global loaded_metrics
        loaded_metrics = load_metric(self.config_dict['metric'])
        # print(loaded_metrics)
        training_args = TrainingArguments(output_dir="../test_trainer",
                                          num_train_epochs=self.config_dict['num_epochs'],
                                          logging_strategy='steps',
                                          per_device_train_batch_size=int(self.config_dict['batch_size']),
                                          per_device_eval_batch_size=int(self.config_dict['batch_size']),
                                          gradient_accumulation_steps=int(self.config_dict['gradient_steps']),
                                          save_steps=int(self.config_dict['save_steps']))

        self.trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            tokenizer=self.tokenizer,
            eval_dataset=tokenized_test,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )
        return


if __name__ == '__main__':
    # Create instance of model
    m = PairwiseModel()
    # Load the configuration file
    m.load_config('config.ini')
    # Load the model specified in the config.ini
    m.load_model(m.config_dict['model_name'])
    # Load the dataset specified in config.ini
    m.load_dataset()

    # Make the dataset smaller
    m.subset_train()
    m.subset_test()

    # Tokenize training set and relabel the outputs
    tokenized_train = m.train_set.map(m.tokenize_input)  # , batched=True)
    tokenized_train = tokenized_train.rename_column(m.config_dict['output'], 'label')
    tokenized_train = tokenized_train.map(m.to_int)
    # tokenized_train = tokenized_train.map(m.map_to_labels)
    # print('STUFF: ', tokenized_train.features)

    # Tokenize test set and relabel the outputs
    tokenized_test = m.test_set.map(m.tokenize_input)  # , batched=True)
    # tokenized_test = tokenized_test.map(m.relabel)  # , batched=False)
    tokenized_test = tokenized_test.rename_column(m.config_dict['output'], 'label')
    tokenized_test = tokenized_test.map(m.to_int)
    # tokenized_test = tokenized_test.map(m.map_to_labels)

    # print(tokenized_test[0])
    # print(tokenized_test)
    # print(tokenized_train[0])

    m.load_trainer(tokenized_train, tokenized_test)
    # m.trainer.train()
    text = """The American Water companies are Aquafina (Pepsi), Dasani (Coke), Perrier (Nestle) which provide jobs for the
         american citizens,[SEP]Americans spend billions on bottled water every year. Banning their sale would greatly 
         hurt an already struggling economy. In addition to the actual sale of uwater bottles, the plastics that they 
         are made out of, and the advertising on both the bottles and packaging are also big bsiness. 
         In addition to this, compostable waters bottle are also coming onto the market, these can be used instead of 
         plastics to eliminate that detriment. Moreover, bottled water not only has a cleaner safety record than 
         municipal water, but it easier to trace when a potential health risk does occur.
        """
    text = m.tokenizer.encode(text, return_tensors='pt').to(device)
    predictions = m.trainer.predict(tokenized_test, ignore_keys=['title', 'stance', 'args', 'reason'])

    # --------------------- DEBUGGING AREA ------------------------ #
    # print(m.total_data, '\n', m.train_set[52301], '\n', m.test_set[0])
    # print(m.tokenizer)
    # print('hi')
    # print(m.trainer)
    # print(tokenized_train[0])
    # print(tokenized_train[100])
    print(type(predictions.predictions), type(predictions.predictions[0]))
    print(len(predictions.predictions), predictions.predictions[0].shape)
    print(predictions.label_ids.shape)
    print(predictions)
    # print(type(predictions))
