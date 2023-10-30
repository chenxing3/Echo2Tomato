import sys, os
import math
import pandas as pd
import numpy as np
import random
import argparse 

from datasets import Dataset
from transformers import (
	GPT2Config,
	GPT2LMHeadModel,
	DataCollatorForLanguageModeling,
	T5Tokenizer,
	Trainer,
	TrainingArguments,
)

def run(args):
	tokenizer = T5Tokenizer(vocab_file=args.token_model_file, use_fast=False) # Load the GPT-2 tokenizer

	# Load the GPT-2 model architecture.
	config = GPT2Config()

	# Modify the configuration attributes as needed
	config.vocab_size = tokenizer.vocab_size  # Set the size of the vocabulary
	config.n_embd = args.model_size  # Set the embedding dimension
	config.n_head = args.num_heads  # Set the number of attention heads
	config.n_layer = args.num_layers  # Set the number of layers
	config.num_labels = tokenizer.vocab_size  # Set the number of labels for classification tasks
	config.activation_function = args.activation_function
	config.use_cache = args.use_cache

	# Create a custom id2label mapping
	config.id2label = {i: label for i, label in enumerate(tokenizer.get_vocab())}
	config.label2id = {label: i for i, label in enumerate(tokenizer.get_vocab())}
	config.pad_token_id = tokenizer.pad_token_id

	# Load the dataset from the step0
	if os.path.exists(args.file_path):
		df = pd.read_feather(args.file_path)
	else:
		print('The file path does not exist: ', args.file_path)
		sys.exit(1)

	texts = [''.join(i.tolist()) for i in df["pattern_ID"].values] # Id for echo pattern
	random.shuffle(texts)

	# encode the dataset
	dataset_encoded = tokenizer(texts, padding='max_length', truncation=True, max_length=args.max_length, return_tensors="pt")

	# split the dataset into training and validation	
	train_dataset = Dataset.from_dict({
		"input_ids": dataset_encoded['input_ids'][:int(len(texts) * args.train_percent)],
		"attention_mask": dataset_encoded['attention_mask'][:int(len(texts) * args.train_percent)],
	})
	train_dataset = train_dataset.shuffle(seed=args.seed) # shuffle the training dataset

	valid_dataset = Dataset.from_dict({
		"input_ids": dataset_encoded['input_ids'][int(len(texts) * args.train_percent):],
		"attention_mask": dataset_encoded['attention_mask'][int(len(texts) * args.train_percent):],
	})

	# print('train_dataset: ', train_dataset[1])
	# sys.exit(1)

	# Create a training arguments object.
	training_args = TrainingArguments(
		output_dir=args.output_dir, # output directory	
		overwrite_output_dir=True,
		num_train_epochs=args.num_epochs, # total number of training epochs
		per_device_train_batch_size=args.batch_size, # batch size per device during training
		per_device_eval_batch_size=32,
		evaluation_strategy= 'epoch',
		logging_strategy= 'epoch',
		weight_decay=args.weight_decay,
		save_strategy= 'epoch',
	)

	model = GPT2LMHeadModel(config) # train from scratch
	# model = GPT2LMHeadModel.from_pretrained(output_dir) # Load the pre-trained GPT-2 model
	model_size = sum(t.numel() for t in model.parameters())
	print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # Create the data collator

	# Create a Trainer instance to train and evaluate the model.
	trainer = Trainer(
		model=model,
		args=training_args,
		data_collator=data_collator,
		train_dataset=train_dataset,
		eval_dataset=valid_dataset,
	)

	# Train the model.
	trainer.train()

	eval_results = trainer.evaluate() # Evaluate the model
	print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

	# Save the model to disk.
	trainer.save_model(args.output_dir) 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Pre-training the GPT-2 model')
	parser.add_argument('--output_dir', default='./result/training_echo', type=str, metavar='PATH',
						help='working directory')
	parser.add_argument('--token_model_file', default='./utils/subword_model_bpe_10000.model', type=str,
						help='the file path of the token model')
	parser.add_argument('--data_file', default='./utils/prototypes_trees_412_IDs.feather', type=str,
						help='the file path of the dataset')
	parser.add_argument('--file_path', default='./result/spectrogram_uniformed.feather', type=str,
						help='the file path of the dataset')
	
	parser.add_argument('--seed', default=31, type=int,
						help='seed for initializing training.')
	parser.add_argument('--max_length', default=180, type=int,
						help='the maximum length of the input sequence')
	parser.add_argument('--train_percent', default=0.9, type=float,
						help='the percentage of training data')
	parser.add_argument('--batch_size', default=64, type=int,
						help='the batch size of training')
	parser.add_argument('--num_epochs', default=60, type=int,
						help='the number of epochs')
	parser.add_argument('--learning_rate', default=5e-5, type=float,
						help='the learning rate')
	parser.add_argument('--weight_decay', default=0.01, type=float,
						help='the weight decay')
	parser.add_argument('--model_size', default=512, type=int,
						help='the model size')
	parser.add_argument('--num_heads', default=8, type=int,
						help='the number of heads')
	parser.add_argument('--num_layers', default=6, type=int,
						help='the number of layers')
	parser.add_argument('--activation_function', default='gelu_new', type=str,
						help='the activation function')
	parser.add_argument('--use_cache', default=True, type=bool,
						help='whether to use cache')

	args = parser.parse_args()


	run(args)
