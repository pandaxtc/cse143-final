"""
Marshal raw messages into data suitable for NMT training.

JoeyNMT takes sentence, translation pairs. This is a little tricky,
because sentence boundaries are really wishy-washy in chats, and a
sentence can be a response to many senteces, so it's not 1:1. Oh well!
"""

import re
import sys
from contextlib import ExitStack
from glob import glob

import numpy as np
import sentencepiece as sp
from nltk import FreqDist, TweetTokenizer, tokenize
# from nnsplit import NNSplit
from numpy.lib.npyio import save
from tqdm import tqdm

from generate_convos import Conversation
from util import load_convos

MODEL_PREFIX = "models/sp_model"

tweet_tokenizer = TweetTokenizer(reduce_len=True)
# nnsplit_tokenizer = NNSplit.load("en")

# def save_messages(out_file):
#     """
#     Save every message to its own line
#     """
#     with open(out_file, "w+") as f:
#         for convo in tqdm(load_convos()):
#             for message in convo.messages:
#                 for c in message.content:
#                     f.write(c + "\n")


# def save_sentences(out_file):
#     """
#     Save only message-sentences to its own line in
#     data/raw_sentences.txt, similar to nnsplit_tokenize in nmt_tokenize
#     """
#     with open(out_file, "w+") as f:
#         for convo in tqdm(load_convos()):
#             for message in convo.messages:
#                 msg = " ".join(message.content)
#                 if message.sender_id == ME:
#                     splits = nnsplit_tokenizer.split([msg])[0]
#                     f.write(str(splits[0]) + "\n")
#                 else:
#                     splits = nnsplit_tokenizer.split([msg])[0]
#                     f.write(str(splits[-1]) + "\n")


def clean(s: str):
    return re.sub(r"http\S+", "<LINK>", s.lower().replace("\n", " "))


def naive_tokenize(convo: Conversation):
    """
    Just concatenates messages into a sentence.
    """
    mine, theirs = [], []

    for i, message in enumerate(convo.messages):
        if i % 2 == 1:
            mine.append(" ".join(map(clean, message.content)))
        else:
            theirs.append(" ".join(map(clean, message.content)))

    if len(theirs) > len(mine):
        assert len(theirs) == len(mine) + 1
        theirs.pop()

    assert len(mine) == len(theirs)
    assert not any(a == "" or b == "" for a, b in zip(mine, theirs))
    return mine, theirs

def load_dataset(data_dir: str, tokenizer=naive_tokenize):
    """
    Load messages dataset into nmt-compatible format.
    """
    mine, theirs = [], []

    for convo in tqdm(load_convos()):
        assert len(convo.participants) == 2

        convo_mine, convo_theirs = tokenizer(convo)
        mine.extend(convo_mine)
        theirs.extend(convo_theirs)

    assert len(mine) == len(theirs)
    mine, theirs = map(np.array, (mine, theirs))

    rng = np.random.default_rng(420)
    indices = rng.permutation(mine.shape[0])

    train, dev, test = np.split(indices, [int(0.7 * len(mine)), int(0.85 * len(mine))])

    train_trg, train_src = mine[train], theirs[train]
    dev_trg, dev_src = mine[dev], theirs[dev]
    test_trg, test_src = mine[test], theirs[test]

    save_tl(data_dir, train_src, train_trg, "train")
    save_tl(data_dir, dev_src, dev_trg, "dev")
    save_tl(data_dir, test_src, test_trg, "test")

    with open(data_dir + "/all.src", "wb+") as out:
        for file in map(lambda f: data_dir + f, ["train.src", "dev.src", "test.src"]):
            with open(file, "rb") as f:
                out.write(f.read())

    with open(data_dir + "/all.trg", "wb+") as out:
        for file in map(lambda f: data_dir + f, ["train.trg", "dev.trg", "test.trg"]):
            with open(file, "rb") as f:
                out.write(f.read())


def save_tl(datadir, source, target, prefix):
    """
    Save src and trg to a dir
    """
    with open(datadir + prefix + ".src", "w+", encoding="utf8") as f:
        for line in source:
            f.write(line + "\n")
    with open(datadir + prefix + ".trg", "w+", encoding="utf8") as f:
        for line in target:
            f.write(line + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        metavar="data_dir",
        type=str,
    )

    args = parser.parse_args()

    load_dataset(args.data_dir)
