import sys, os
import math
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    confusion_matrix
)

from torch import nn
from torch.nn import Identity
from datasets import Dataset
from transformers import (
    GPT2Config, 
    TrainingArguments, 
    T5Tokenizer, 
    Trainer, 
    AutoModelForSequenceClassification
)

# from collections import Counter

class Identity(nn.Module):  # This is a custom identity module that lets us pass the logits directly to the loss function
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x
	

def re_assign_label(labels, cuts):
	'''
	For to re-assign the label

	*input: 
	labels: the original labels
	cuts:   the cut points for the new labels

	*output:
	new_labels: the new labels
	'''
	labels = np.array(labels)

	new_labels = np.arange(0, len(cuts))
	for idx in range(len(cuts)):
		if idx == 0:
			labels[labels==idx]= new_labels[idx]
		else:
			labels[np.logical_and(labels>cuts[idx-1], labels<= cuts[idx])]= new_labels[idx]
	
	# counts = Counter(labels)
	# print('counts: ', counts)
	# sys.exit(1)
	return labels.tolist()

def calculate_accuracy(predicted_labels, target_labels, buffer):
	'''
	this function is used to calculate the accuracy within a buffer of +/- 1, 2, and 3

	*input:
	predicted_labels: the predicted labels
	target_labels: the target labels
	buffer: the buffer

	*output:
	accuracy: the accuracy within a buffer of +/- 1, 2, and 3
	'''
	correct_predictions = 0
	total_predictions = len(predicted_labels)

	for i in range(total_predictions):
		lower_bound = target_labels[i] - buffer
		upper_bound = target_labels[i] + buffer

		if lower_bound <= predicted_labels[i] <= upper_bound:
			correct_predictions += 1

	accuracy = (correct_predictions / total_predictions) * 100
	return accuracy

def plotting(cm, epoch, rate, buffer_1, buffer_2, buffer_3):
	'''
	For to plot the confusion matrix
	
	*input:
	cm: the confusion matrix
	epoch: the epoch number
	rate: the accuracy
	buffer_1: the accuracy within a buffer of +/- 1
	buffer_2: the accuracy within a buffer of +/- 2
	buffer_3: the accuracy within a buffer of +/- 3

	'''

	# Create a figure and axes
	fig, ax = plt.subplots(figsize=(10, 10))
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)

	# Set the labels for x and y axes
	label_changed = [0, 5, 7, 8, 9, 10, 11, 12, 14, 16, 18, 21, '>21']
	ax.set(xticks=np.arange(cm.shape[1]),
			yticks=np.arange(cm.shape[0]),
			xticklabels=label_changed,
			yticklabels=label_changed,
			xlabel='Predicted label',
			ylabel='True label',
			title='Confusion Matrix: '+str(np.round(rate*100, 1)) + '%, ' + str(buffer_1) + '%, ' + str(buffer_2) + '%, ' + str(buffer_3) + '%'
			)

	# Rotate the x-axis labels for better readability (if needed)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > np.max(cm) / 2.0 else "black")

	# Save the figure
	plot_file = './result/confusion_matrix/confusion_matrix_'+str(int(epoch)).zfill(2) +'.png'
	plt.savefig(plot_file)
	plt.close()


def compute_metrics(eval_pred, epoch):
	'''
	this function is used to compute the metrics

	*input:
	eval_pred: the evaluation prediction
	epoch: the epoch number

	*output:
	metric_res: the metrics result
	'''

	logits, labels = eval_pred # Get the predictions and labels
	predictions = np.argmax(logits, axis=-1) # Get the predictions

	accuracy = accuracy_score(labels, predictions)
	recall = recall_score(labels, predictions, average='macro')  # Use 'macro' for multiclass data
	precision = precision_score(labels, predictions, average='macro')  # Use 'macro' for multiclass data
	f1 = f1_score(labels, predictions, average='macro')  # Use 'macro' for multiclass data

	# Calculate the accuracy within a buffer of +/- 1, 2, and 3
	buffer_1 = calculate_accuracy(predictions, labels, 1) # buffer = 1
	buffer_2 = calculate_accuracy(predictions, labels, 2) # buffer = 2
	buffer_3 = calculate_accuracy(predictions, labels, 3) # buffer = 3

	metric_res = {
		"accuracy": accuracy,
		"recall": recall,
		"precision": precision,
		"f1": f1,
		'+-1': buffer_1,
		'+-2': buffer_2,
		'+-3': buffer_3,
	}

	cm = confusion_matrix(labels, predictions) # Create a confusion matrix
	plotting(cm, epoch, accuracy, round(buffer_1, 1), round(buffer_2, 1), round(buffer_3, 1)) # Plot the confusion matrix	
	return metric_res

def run(args):

	tokenizer = T5Tokenizer(vocab_file=args.token_model_file, use_fast=False) # Load the GPT-2 tokenizer

	# Load the GPT-2 model architecture.
	config = GPT2Config()

	# Modify the configuration attributes as needed
	config.vocab_size = tokenizer.vocab_size  # Set the size of the vocabulary
	config.n_embd = args.model_size  # Set the embedding dimension
	config.n_head = args.num_heads # Set the number of attention heads
	config.n_layer = args.num_layers  # Set the number of layers
	config.num_labels = tokenizer.vocab_size  # Set the number of labels for classification tasks
	config.activation_function = args.activation_function  # Set the activation function
	# config.report_to = 'tensorboard'
	config.use_cache = args.use_cache # Set to True to speed up training by reusing computation

	# Create a custom id2label mapping
	config.id2label = {i: label for i, label in enumerate(tokenizer.get_vocab())}
	config.label2id = {label: i for i, label in enumerate(tokenizer.get_vocab())}
	config.pad_token_id = tokenizer.pad_token_id

	# Load the dataset
	df = pd.read_feather(args.file_path)

	texts = [''.join(i.tolist()) for i in df["pattern_ID"].values] # Id for echo pattern
	# tokens = [tokenizer.encode(''.join(i.tolist()), return_tensors="np")[0].tolist() for i in df["pattern_ID"].values]
	labels = np.array([int(i) for i in df['fruit_number'].values])

	# since the inbalanced data, we need to re-assign the label
	cuts = [0, 5, 7, 8, 9, 10, 11, 12, 14, 16, 19, 21, max(labels)+1] # it get new labels with balanced number: 5: 1888, 4: 1752, 6: 1388, 2: 1312, 3: 1180, 7: 1032, 0: 976, 8: 920, 11: 892, 1: 744, 9: 716, 10: 460
	labels = re_assign_label(labels, cuts)

	# Create a list of indices
	indices = list(range(len(texts)))

	# Shuffle the indices to rearrange the data
	random.shuffle(indices)

	# Use the shuffled indices to access data and create datasets
	shuffled_texts = [texts[i] for i in indices]
	shuffled_labels = [labels[i] for i in indices]

	# encode the dataset
	dataset_encoded = tokenizer(shuffled_texts, padding='max_length', truncation=True, 
							 	max_length=args.max_length, return_tensors="pt")

	train_percent = args.train_percent
	train_indices = indices[:int(len(texts) * train_percent)]
	valid_indices = indices[int(len(texts) * train_percent):]

	train_dataset = Dataset.from_dict({
		"input_ids": dataset_encoded['input_ids'][train_indices],
		"attention_mask": dataset_encoded['attention_mask'][train_indices],
		"labels": [shuffled_labels[i] for i in train_indices]
	})

	valid_dataset = Dataset.from_dict({
		"input_ids": dataset_encoded['input_ids'][valid_indices],
		"attention_mask": dataset_encoded['attention_mask'][valid_indices],
		"labels": [shuffled_labels[i] for i in valid_indices]
	})

	# print('train_dataset: ', train_dataset[1])
	# sys.exit(1)

	# Create a training arguments object.
	model_name = args.output_dir.split("/")[-1]

	training_args = TrainingArguments(
		output_dir=f"{model_name}-finetuned-echo",
		num_train_epochs= args.num_epochs,
		per_device_train_batch_size= args.batch_size,
		per_device_eval_batch_size=32,
		evaluation_strategy= 'epoch',
		logging_strategy= 'epoch',
		weight_decay=args.weight_decay,
		save_strategy= 'epoch',
		label_smoothing_factor=0.3,
	)

	num_labels = len(set(labels))
	print('num_labels: ', num_labels)
	# model = AutoModelForSequenceClassification.from_config(config) # run from scratch
	model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)

	model.lm_head = Identity() # Set the LM head to identity

	# Add a classification head	
	model.classifier = nn.Sequential(
		nn.Linear(config.n_embd, 256),
		nn.ReLU(),
		nn.Dropout(0.1),
		nn.Linear(256, num_labels)
	)
	
	model_size = sum(t.numel() for t in model.parameters())
	print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

	# Create a Trainer instance to train and evaluate the model.
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=valid_dataset,
		compute_metrics=lambda eval_pred: compute_metrics(eval_pred, trainer.state.epoch),  # Pass the epoch number
	)

	# Train the model.
	trainer.train()

	eval_results = trainer.evaluate()
	print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

	# Save the model to disk.
	trainer.save_model(f"{model_name}-finetuned-echo")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Fine-tune the GPT-2 model')
	
	parser.add_argument('--output_dir', default='./result/training_echo', type=str, metavar='PATH',
						help='working directory')
	parser.add_argument('--token_model_file', default='./utils/subword_model_bpe_10000.model', type=str,
						help='the file path of the token model')
	parser.add_argument('--data_file', default='./utils/prototypes_trees_412_IDs.feather', type=str,
						help='the file path of the dataset')
	parser.add_argument('--file_path', default='./result/spectrogram_uniformed.feather', type=str,
						help='the file path of the dataset')
	

	parser.add_argument('--max_length', default=180, type=int,
						help='the maximum length of the input sequence')
	parser.add_argument('--train_percent', default=0.8, type=float,
						help='the percentage of training data')
	parser.add_argument('--batch_size', default=96, type=int,
						help='the batch size of training')
	parser.add_argument('--num_epochs', default=50, type=int,
						help='the number of epochs')
	parser.add_argument('--learning_rate', default=5e-5, type=float,
						help='the learning rate')
	parser.add_argument('--weight_decay', default=0.01, type=float,
						help='the weight decay')
	parser.add_argument('--model_size', default=256, type=int,
						help='the model size')
	parser.add_argument('--num_heads', default=8, type=int,
						help='the number of heads')
	parser.add_argument('--num_layers', default=6, type=int,
						help='the number of layers')
	parser.add_argument('--activation_function', default='gelu_new', type=str,
						help='the activation function')
	parser.add_argument('--label_smoothing_factor', default=0.3, type=float,
						help='the label smoothing factor')
	parser.add_argument('--use_cache', default=True, type=bool,
						help='whether to use cache')
	
	args = parser.parse_args()
	run(args)
