import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import hashing_trick
import struct
import socket
import glob
import multiprocessing.pool as mpp
import tqdm


def createDocFromFile(filename, sampling_rate=None):
    """
    Creates a json doc from the CSV file
    """
    domain = filename.split("/")[-2]
    UA = filename.split("/")[-4]
    df = pd.read_csv(filename)
    #df.serverIP = df.serverIP.apply(str2IP)
    dns_server = filename.split(
        "/")[0].replace("pcaps_dns_", "").replace("_", ".")
    if sampling_rate is not None:
        all_packets = df.totalPackets.sum()
        probabilities = df.totalPackets / all_packets
        s = all_packets * sampling_rate
        selected_indices = set(np.random.choice(
            df.index, size=s, p=probabilities))
        filtered = df.iloc[selected_indices, :]
    else:
        filtered = df

    return {"words": filtered.serverIP.values, "number": np.uint32(filtered.serverIP.apply(str2IP).values), "tags": [domain], "filename": filename, "UA": UA, "dns_server": dns_server, "totalPackets": filtered.totalPackets, "totalBytes": filtered.totalBytes}


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


vocab_length = 60000
max_length = 250

def load_dataset(basedir="~/shared-nvme/datos_IPs", vocab_length=vocab_length, max_length=max_length, sampling_rate=None, parallel=False):
    # Create corpus
    def load_file(filename):
        return createDocFromFile(filename, sampling_rate=sampling_rate)
    if parallel:
        with mpp.ThreadPool(128*2) as pool:
            corpus = pool.map(load_file, tqdm.tqdm(glob.glob(f"{basedir}/pcaps_dns_*/U*/PartialFootprints/*/*.summary")))
    else:
        corpus = [createDocFromFile(filename, sampling_rate=sampling_rate) for filename in tqdm.tqdm(glob.glob(f"{basedir}/pcaps_dns_*/U*/PartialFootprints/*/*.summary"))]

    # Build vector Y
    classes = [e["tags"][0] for e in corpus]
    s = [e["dns_server"] for e in corpus]
    # Build one-hot encoding
    embedded_sentences = [hashing_trick(
        " ".join(e["words"]), vocab_length, hash_function="md5") for e in corpus]
    # Build padded sentences
    padded_sentences = pad_sequences(
        embedded_sentences, max_length, padding='post')

    return padded_sentences, s, classes, embedded_sentences, corpus


data = load_dataset()

with open("data.pkl", "wb") as f:
    f.write(pickle.dumps(data))
