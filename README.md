# Echo2Tomato
## Determining Tomato Count via Robot Echoes
In this project, I developed a GPT-2 model designed to detect specific frequency sequences within spectrograms. These spectrograms were generated from echo waveforms, which consisted of both transmitted and received signal components. To standardize the signal components, I employed the 'convolve' method, using an ideal chirp as a reference. The standardized chirps were then converted into spectrograms with time-varying frequency patterns. To identify these frequency patterns, I applied hierarchical cluster analysis using Ward's method.


# Dependencies
transformers (support both pytorch and tensorflow: https://github.com/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)

datasets

pandas

numpy

scipy

sklearn

tqdm

argparse

matplotlib

torch (Since the I add 'classification' layers using pytorch. If using Tensorflow, please rewrite class of Identity and Classify layer (detail in step2 in below)


# Executing
## Step1 convert echo to spectrogram

To begin this process, please run the Step0_preprocess_spectrogram.py script. In this step, I initially defined the ideal chirp. Subsequently, I standardized the echo waveforms and transformed them into spectrograms using the Short-Time Fourier Transform (STFT). The frequency patterns within these spectrograms were mapped to unique characters, a crucial step in the subsequent tokenization process. For your convenience, I have provided a total of 417 spectrum patterns, referred to as prototypes, which encompass all the patterns in our database. These prototype patterns are stored in the file located at ./utils/prototypes_trees_412_IDs.feather.

    $ python Step0_preprocess_spectrogram.py 

* Please move the all the feather data to 'data' folder or specify the data directory using following command:

    $ python Step0_preprocess_spectrogram.py --directory=/your_data_path/

* the script collects all the data in the file './result/spectrogram_uniformed.feather'.

## Step2 Model Training
To proceed, please run the Step1_training.py script. In this step, the frequency patterns were transformed into sequences, which were subsequently tokenized using the 'subword_model_bpe_10000.model' that was created using Byte-Pair Encoding (BPE) methodology. For pretraining the data, I employed the 'GPT2LMHeadModel' to facilitate the training process.

    $ python Step1_training.py


## Step 3: Fine-Tuning the Model
I utilized 'AutoModelForSequenceClassification' for fine-tuning the model. Typically, it requires approximately 5 to 10 epochs to achieve the best model performance. It's worth noting that fine-tuning yields  better results compared to training the model from scratch (using the model in line 254).

    $ python Step2_fine_tune.py

* If using tensorflow, please edit "classify layer" in lines 257-265, and "identity layer" in lines "30-35"


## Results:
![confusion_matrix_07](https://github.com/chenxing3/Echo2Tomato/assets/20653768/8859e509-03f6-4f02-aa6d-231a330c84ac)

the rates in title is accuracy rate and with the buffer of +/- 1, 2, and 3. I think if more data, the result would be better. 
