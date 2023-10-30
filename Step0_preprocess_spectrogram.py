import sys, os
import argparse
import pandas as pd 
import numpy as np 
# import glob
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial import cKDTree


def find_feather_files(directory):
	'''
	Function to find all Feather files in a directory and its subdirectories
	'''
	feather_files = []

	for root, dirs, files in os.walk(directory):
		for file in files:
			if file.endswith('.feather'):
				feather_files.append(os.path.join(root, file))

	return feather_files


def get_ideal_chirp(sample_rate):
	'''
	For to generate the idea chirp
	'''
	start_frequency = 21000  # Hz  # it was tested the best one for the field data
	end_frequency = 100000  # Hz
	chirp_duration = 0.005
	t = np.linspace(0, chirp_duration, int(chirp_duration * sample_rate), endpoint=False)
	matched_filter = signal.chirp(t, start_frequency, chirp_duration, end_frequency)
	return matched_filter

def get_spectrogram(echo_vector, matched_filter, freq_start, freq_end, args):
	'''
	For to get the spectrogram of the target chirp

	*input: 
	echo_vector:        the echo oscilloscope data
	matched_filter:     the idea chirp
	sample_rate:        the sample rate of the echo_vector
	nperseg:            the window size of the spectrogram
	nfft:               the length of the FFT used
	noverlap:           the number of points of overlap between segments
	ref_start:          the start index of the transmitted chirp
	ref_end:            the end index of the transmitted chirp
	freq_start:         the start index of the target frequency
	freq_end:           the end index of the target frequency
	target_chirp_start: the start index of the target chirp
	target_chirp_end:   the end index of the target chirp

	*output:
	selected_spectrogram_uniformed: the uniformed spectrogram of the target chirp
	'''

	# uniform the received chirp by dividing the transmitted chirp
	filtered_signal = signal.convolve(echo_vector, matched_filter, mode='same') 
	f, t, z = signal.stft(filtered_signal, args.sample_rate, nperseg=args.nperseg, nfft=args.nfft, noverlap=args.noverlap)
	z = np.abs(z) # get absolute value

	# selected transmitted chirp
	transmitted_spectrogram = z[int(len(f)*freq_start)+1:int(len(f)*freq_end), args.ref_start:args.ref_end]
	transmitted_spectrogram_mean = np.mean(transmitted_spectrogram, axis=1)

	# selected target ranges of frequecy and time axes
	selected_spectrogram = z[int(len(f)*freq_start)+1:int(len(f)*freq_end), args.target_chirp_start:args.target_chirp_end] 

	# uniformed the signal by dividing the transmitted chirp
	selected_spectrogram_uniformed = selected_spectrogram/transmitted_spectrogram_mean[:,None]

	# remove the noise (when value > median*2)
	selected_spectrogram_uniformed[selected_spectrogram_uniformed < np.median(np.mean(selected_spectrogram_uniformed, axis=0)*2)] = 0

	return selected_spectrogram_uniformed


def get_reference_tree(file):
	'''
	For to get the reference tree which is convenient to find the nearest spectrum pattern for each echo

	*input:
	file: the file path of the reference tree

	*output:
	tree: the reference tree
	IDs: the ID of the spectrum pattern, the ID is some unique character, includes english aplhabets and some chinese characters
	prototypes: the vector of the spectrum pattern
	'''

	df_IDs = pd.read_feather(file)
	prototypes = []
	IDs = []

	for i in range(len(df_IDs)):
		prototype = df_IDs['tree_prototypes'][i].tolist()
		ID = df_IDs['ID'][i]
		prototypes.append(prototype)
		IDs.append(ID)

	prototypes = np.array(prototypes)
	tree = cKDTree(prototypes)
	return tree, IDs, prototypes


def get_uniformed_data(args):
	# Step1 prepare idea chirp which used in the tomato data in the field
	idea_chirp = get_ideal_chirp(args.sample_rate)

	# The sample rate is the target frequency ranges from 30kHZ to 100kHZ
	## if set nfft to 256, the spectrum matrix is selected from freq/(sample_rate/2)
	freq_start = args.start_frequency/(args.sample_rate/2)
	freq_end = args.end_frequency/(args.sample_rate/2)

	# echoes have two parts: transmitted chirp and received chirp
	## the transmitted chirp start from 92*4 to 98*4 if the spectrogram generated using window size=8 and overlap=3 (nfft=256)
	## the most received chirp distributed from 320 to 1100 (about 4 ms to 11 ms)

	df_all_data = pd.DataFrame() # save all the data

	feather_files = find_feather_files(args.directory) # find all the feather files in the directory

	if len(feather_files) == 0:
		print('No feather files found in the directory')
		sys.exit(1)

	print('-------> Convert to spectrogram ....')
	for file in tqdm(feather_files):
		df = pd.read_feather(file)

		# get amplitude to uniform the echos
		amplitude = df['amp'][0]
		echoes = df['echos'][0]/amplitude
		
		# there are four echoes for each file, and we preprocess according to the idea chirp
		## the preprocess has three operations: 
		### Step1: get the transmitted chirp and uniform the received chirp by dividing the transmitted chirp
		### Step2: get the spectrogram of the uniformed received chirp
		### Step3: remove the noise in the spectrogram
		spectrograms_uniformed = []
		for echo in echoes:
			if echo.shape != (8000, ): # some echos are not 8000, so we skip them
				continue

			spectrogram_uniformed = get_spectrogram(echo, idea_chirp,freq_start, freq_end,  args)

			spectrograms_uniformed.append(spectrogram_uniformed.T.tolist())

		df_spectrogram = pd.DataFrame({'spectrogram_uniformed':spectrograms_uniformed, 
								 	   'fruit_number': [df['fruit'].values[0]]*len(spectrograms_uniformed),
									   'file':[os.path.basename(file)]*len(spectrograms_uniformed)})
		
		# concat df_all_data and df_spectrogram
		df_all_data = pd.concat([df_all_data, df_spectrogram], ignore_index=True)

	# add echo index
	## The index is corresponding to spectrum pattern in frequency domain. I found 412 patterns in our dataset (collected from the two tomato fields)
	## I adopted KD-tree to build the prototype tree which is convenient to find the nearest spectrum pattern for each echo
	print('-------> Add echo index....')
	tree, IDs, prototypes = get_reference_tree(args.prototypes_file) # all the 412 spectrum vectors with the size of 256 dimension

	pattern_IDs = []
	for i in tqdm(range(len(df_all_data))):
		echo_spectrogram = df_all_data['spectrogram_uniformed'][i] # get the spectrogram of each echo

		ID_vector = []
		for spectrogram in echo_spectrogram:
			query = np.array(spectrogram).reshape(1, -1) # create a query vector
			_, ind = tree.query(query, k=1) # find the nearest spectrum pattern
			ID_vector.append(IDs[ind[0]])
		
		pattern_IDs.append(ID_vector)

	df_all_data['pattern_ID'] = pattern_IDs
	file_save = args.output_file + 'spectrogram_uniformed.feather'
	df_all_data.to_feather(file_save)

	if os.path.exists(file_save):
		print('The uniformed spectrogram is saved in: ', file_save)
		print('Done!')
	else:
		print('Error! failed to save the uniformed spectrogram')
		sys.exit(1)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Preprocess echo data to get the uniformed spectrogram')
	parser.add_argument('--directory', default='./data/', type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')
	parser.add_argument('--output_file', default='./result/', type=str,
						help='the file path of the output file')
	parser.add_argument('--prototypes_file', default='./utils/prototypes_trees_412_IDs.feather', type=str,
						help='the file path of the reference tree')
	
	parser.add_argument('--sample_rate', default=500e3, type=int,
						help='the sample rate of the echo_vector')
	parser.add_argument('--start_frequency', default=30000, type=int,
						help='the start frequency of the target chirp')
	parser.add_argument('--end_frequency', default=100000, type=int,
						help='the end frequency of the target chirp')
	parser.add_argument('--nfft', default=256, type=int,
						help='the length of the FFT used')
	parser.add_argument('--ref_start', default=92*4, type=int,
						help='the start index of the transmitted chirp')
	parser.add_argument('--ref_end', default=98*4, type=int,
						help='the end index of the transmitted chirp')
	parser.add_argument('--nperseg', default=8, type=int,
						help='the window size of the spectrogram')
	parser.add_argument('--noverlap', default=3, type=int,
						help='the number of points of overlap between segments')
	parser.add_argument('--target_chirp_start', default=320, type=int,
						help='the start index of the target chirp')
	parser.add_argument('--target_chirp_end', default=1100, type=int,
						help='the end index of the target chirp')

	args = parser.parse_args()
	get_uniformed_data(args)


