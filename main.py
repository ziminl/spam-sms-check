import time
import pickle
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)

import tqdm
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall, Precision


SEQUENCE_LENGTH = 100 
EMBEDDING_SIZE = 100 
TEST_SIZE = 0.25

BATCH_SIZE = 64
EPOCHS = 10 

label2int = {"o": 0, "x": 1}
int2label = {0: "o", 1: "x"}


def load_data():
    """
    loads spam collection dataset
    """
    texts, labels = [], []
    with open("data/SMSSpamCollection") as f:
        for line in f:
            split = line.split()
            labels.append(split[0].strip())
            texts.append(' '.join(split[1:]).strip())
    return texts, labels
    
X, y = load_data()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
pickle.dump(tokenizer, open("results/tokenizer.pickle", "wb"))
X = tokenizer.texts_to_sequences(X)
