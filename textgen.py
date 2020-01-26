import os
import tensorflow as tf
from tensorflow import keras
from textgenrnn import textgenrnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
textgen = textgenrnn(weights_path='new_model_weights.hdf5',
                       vocab_path='new_model_vocab.json',
                       config_path='new_model_config.json')

text = textgen.train_from_file('beatleslyrics.txt',
                        new_model=False,
                        rnn_bidirectional=True,
                        rnn_size=64,
                        dim_embeddings=300,
                        num_epochs=10)
