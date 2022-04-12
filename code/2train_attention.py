import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder 
import numpy as np
from tensorflow.keras.layers import *
import tensorflow.keras.layers as layers
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import *
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import hashing_trick
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
import datetime
import xxhash
from lib import *
import hashlib

parser = argparse.ArgumentParser(description='Simple model training')
parser.add_argument('-v','--vocab_length', help='Length of the vocabulary', default=60000, type=int)
parser.add_argument('-m','--max_length', help='Length of the sequence', default=250, type=int)
parser.add_argument('-e','--n_epochs', help='Number of epochs', default=1000, type=int)
parser.add_argument('-l','--log_file', help='Log file', required=True, type=str)
parser.add_argument('-H','--hash_function', help='Hash function', required=True, type=str)
parser.add_argument('-SM','--save-model', help='Save best model', required=False, type=str)
parser.add_argument('-LM','--load-model', help='Load model to continue training', required=False, type=str)
parser.add_argument('-val','--use-validation', help='Use validation for CSV logger', action="store_true")
parser.add_argument('-s','--sampling-rate', help='Sampling rate for training', required=False, type=float, default=None)
parser.add_argument('-sa', "--sampling-additional-samples", help="Number of additional samples when sampling", required=False, type=int, default=5)

args = vars(parser.parse_args())

max_length, vocab_length = args["max_length"], args["vocab_length"]
n_epochs, log_file  = args["n_epochs"], args["log_file"]
hash_function = args["hash_function"]
save_model, load_model = args["save_model"], args["load_model"]
use_validation = args["use_validation"]
sampling_rate = args["sampling_rate"]
sampling_add_samples = args["sampling_additional_samples"]

def clock():
    return datetime.datetime.now().timestamp()

def xxh32(e):
    if e == 0:
        return 0
    if type(e) == int:
        e = e.to_bytes(4, "little")
    else:
        e = e.tobytes()

    return (xxhash.xxh32(e).intdigest() % vocab_length) 

def xxh64(e):
    if e == 0:
        return 0
    if type(e) == int:
        e = e.to_bytes(4, "little")
    else:
        e = e.tobytes()

    return (xxhash.xxh64(e).intdigest() % vocab_length)

def md5(e):
    if e == 0:
        return 0
    if type(e) == int:
        e = e.to_bytes(4, "little")
    else:
        e = e.tobytes()

    return int(hashlib.md5(e).hexdigest(), 16) % vocab_length

def sha1(e):
    if e == 0:
        return 0
    if type(e) == int:
        e = e.to_bytes(4, "little")
    else:
        e = e.tobytes()

    return int(hashlib.sha1(e).hexdigest(), 16) % vocab_length

hashed = {}
counter = 1
TRAINING=True
def nohash(e):
    global hashed
    global counter
    global TRAINING
    if e == 0:
        return 0
    
    if e in hashed:
        return hashed[e]
    else:
        if TRAINING:
            hashed[e] = counter
            counter += 1
            return hashed[e]
        else:
            return 0
    
if hash_function == "xxh32":
    h = np.vectorize(xxh32)
elif hash_function == "xxh64":
    h = np.vectorize(xxh64)
elif hash_function == "md5":
    h = np.vectorize(md5)
elif hash_function == "sha1":
    h = np.vectorize(sha1)
elif hash_function == "nohash":
    h = lambda e: np.asarray([nohash(i) for i in e])
print("Parameters")
print(f"vocab_length={vocab_length}")
print(f"max_length={max_length}")

print("Loading data")
with open("data.pkl", "rb") as f:
    X, _, classes, embedded_sentences, corpus = pickle.loads(f.read())

    
if use_validation:
    train_corpus, test_corpus = train_test_split(corpus, test_size=0.1, stratify=classes, random_state=0)
    corpus = train_corpus
    classes = [e["tags"][0] for e in corpus]

TOTALPACKETS_PRESENT = "totalPackets" in corpus[0] 

if sampling_rate is not None:
    print(f"Applying sampling s={sampling_rate}")
    modified_corpus = []
    modified_classes = []
    for d in corpus:
        for i in range(sampling_add_samples):
            if TOTALPACKETS_PRESENT:    
                all_packets = d["totalPackets"].sum()
                probabilities = d["totalPackets"] / all_packets
                s = int(all_packets * sampling_rate)
                selected_indices = list(set(np.random.choice(range(d["totalPackets"].shape[0]), size=s, p=probabilities)))
                filtered = d["number"][selected_indices]
                doc =  {"number": np.uint32(filtered), "tags": d["tags"], "filename": d["filename"], "UA": d["UA"], "dns_server": d["dns_server"]}
                if doc["number"].shape[0] == 0:
                    print(d, doc, s)
            else:            
                df = pd.read_csv(d["filename"])
                all_packets = df.totalPackets.sum()
                probabilities = df.totalPackets / all_packets
                s = int(all_packets * sampling_rate)
                selected_indices = list(set(np.random.choice(
                    df.index, size=s, p=probabilities)))
                filtered = df.iloc[selected_indices, :]
                doc =  {"words": filtered.serverIP.values, "number": np.uint32(filtered.serverIP.apply(str2IP).values), "tags": d["tags"], "filename": d["filename"], "UA": d["UA"], "dns_server": d["dns_server"], "totalPackets": filtered.totalPackets, "totalBytes": filtered.totalBytes}
            modified_corpus.append(doc)
            modified_classes.append(d["tags"][0])

    classes = modified_classes
    corpus = modified_corpus

# Build one-hot encoding
embedded_sentences = [h(e["number"]) for e in corpus]
TRAINING=False
if hash_function == "nohash":
    vocab_length = counter
    print(f"New vocab length={vocab_length}")
# Build padded sentences
padded_sentences = pad_sequences(embedded_sentences, max_length, padding='post')
X = padded_sentences

print("Splitting data")
nclasses = np.unique(classes).shape[0]
class_encoder = LabelEncoder()
class_encoder.fit(sorted(np.unique(classes)))
Y = to_categorical(class_encoder.fit_transform(classes))

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=1)


print("Creating the model")
if load_model:
    model = keras.models.load_model(load_model, custom_objects={"TokenAndPositionEmbedding": TokenAndPositionEmbedding, "TransformerBlock": TransformerBlock})
else:
    embed_dim = 32  # 64 # Embedding size for each token
    num_heads = 8  # 8 #Number of attention heads
    ff_dim = 32  # 64 # Hidden layer size in feed forward network inside transformer
    inputs = layers.Input(shape=(max_length,))
    x = TokenAndPositionEmbedding(max_length, vocab_length, embed_dim, position_embedding=False)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(nclasses, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[CategoricalAccuracy()])
print(model.summary())
print("Starting training process")

csv_logger = CSVLogger(log_file)
callbacks = [csv_logger]
if save_model:
    mcp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model, monitor='val_categorical_accuracy', mode='max', save_best_only=True)
    callbacks += [mcp]
model.fit(X_train, Y_train, epochs=n_epochs, batch_size=256, verbose=1, callbacks=callbacks, validation_data=(X_val, Y_val))

