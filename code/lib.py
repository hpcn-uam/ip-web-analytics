import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import xxhash
import hashlib
import struct
import socket

def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]


def str2IP(addr):
    addr2 = addr.split(",")
    if len(addr2) > 1:
        addr = addr.replace(",172.17.0.3", "")
    pieces = addr.split(".")
    if len(pieces) != 4:
        print(addr)
    return ip2int(addr)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, position_embedding=True, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.position_embedding = position_embedding
        
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        if self.position_embedding:
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        x = self.token_emb(x)
        if self.position_embedding:
            maxlen = tf.shape(x)[-1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            return x + positions
        else:
            return x
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'position_embedding': self.position_embedding,
        })
        return config
    
        
def clock():
    return datetime.datetime.now().timestamp()

class CSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, file):
        self.file = file
        self.f = open(self.file, "w")
        self.initialized = False
    def on_epoch_end(self, epoch, logs=None):
        if self.initialized == False:
            self.keys = list(logs.keys())
            header = ["#epochs", "ts"] + self.keys
            self.f.write(",".join(header)+"\n")
            self.initialized = True
            
        data = [f"{epoch}", f"{clock()}"]
        for k in self.keys:
            data += [f"{logs[k]}"] if k in logs else None
        self.f.write(",".join(data)+"\n")
        if epoch % 5:
            self.f.flush()
        
    def on_train_end(self, logs=None):
        self.f.close()
        
        