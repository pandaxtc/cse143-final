"""
Marshal raw messages into data suitable for NMT training.

JoeyNMT takes sentence, translation pairs. This is a little tricky,
because sentence boundaries are really wishy-washy in chats, and a
sentence can be a response to many senteces, so it's not 1:1. Oh well!
"""

import re
import sys

import numpy as np
import sentencepiece as sp
from nltk import FreqDist, TweetTokenizer, tokenize
from nnsplit import NNSplit
from numpy.lib.npyio import save
from tqdm import tqdm
from contextlib import ExitStack
from glob import glob

from convo import ME, Conversation
from util import load_convos

SP_TRAINING_FILE = "data/nmt/messages_nnsplit/"
MODEL_PREFIX = "models/sp_model"
OUT_DIR = "data/nmt/messages_nnsplit/"

tweet_tokenizer = TweetTokenizer(reduce_len=True)
nnsplit_tokenizer = NNSplit.load("en")

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

    for message in convo.messages:
        if message.sender_id == ME:
            if len(theirs) == 0:
                continue
            mine.append(
                " ".join(
                    tweet_tokenizer.tokenize(" ".join(map(clean, message.content)))
                )
            )
        else:
            theirs.append(
                " ".join(
                    tweet_tokenizer.tokenize(" ".join(map(clean, message.content)))
                )
            )

    if len(theirs) > len(mine):
        assert len(theirs) == len(mine) + 1
        theirs.pop()

    assert len(mine) == len(theirs)
    assert not any(a == "" or b == "" for a, b in zip(mine, theirs))
    return mine, theirs


def nnsplit_tokenize(convo: Conversation):
    """
    Uses NNSplit to get only the last sentence of the source and the
    first sentence of the target.
    """
    mine, theirs, my_vocab, their_vocab = [], [], set(), set()
    vocab = FreqDist()

    for message in convo.messages:
        words = tweet_tokenizer.tokenize(" ".join(map(clean, message.content)))
        msg = " ".join(words)

        if message.sender_id == ME:
            if len(theirs) == 0:
                continue

            for word in words:
                vocab[word] += 1
                my_vocab.add(word)

            splits = nnsplit_tokenizer.split([msg])[0]
            mine.append(str(splits[0]))
        else:
            for word in words:
                vocab[word] += 1
                their_vocab.add(word)

            splits = nnsplit_tokenizer.split([msg])[0]
            theirs.append(str(splits[-1]))

    if len(theirs) > len(mine):
        assert len(theirs) == len(mine) + 1
        theirs.pop()

    # vocab = my_vocab | their_vocab

    assert len(mine) == len(theirs)
    assert not any(a == "" or b == "" for a, b in zip(mine, theirs))
    return mine, theirs, vocab


def load_dataset(data_dir: str, tokenizer=nnsplit_tokenize):
    """
    Load messages dataset into nmt-compatible format.
    """
    mine, theirs, vocab = [], [], set(["<unk>", "<pad>", "<s>", "</s>"])
    vocab = FreqDist()

    for convo in tqdm(load_convos()):
        assert ME in convo.participants.values(), str(convo)

        convo_mine, convo_theirs, new_vocab = nnsplit_tokenize(convo)
        mine.extend(convo_mine)
        theirs.extend(convo_theirs)
        # vocab |= new_vocab
        vocab += new_vocab

    # percentile = int(0.2 * len(vocab))
    # vocab = [c[0] for c in vocab.most_common(percentile)]
    # vocab = list(vocab)
    # vocab.sort()
    # save_vocab(data_dir, vocab)

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

    with open(data_dir + "/all", "rb") as out:
        for file in filter(glob(data_dir + "/*")):
            with open(file, "rb") as f:
                out.write(f.read())

    with open(data_dir + "/all.src", "wb+") as out:
        for file in map(lambda f: data_dir + f, ["train.src", "dev.src", "test.src"]):
            with open(file, "rb") as f:
                out.write(f.read())

    with open(data_dir + "/all.trg", "wb+") as out:
        for file in map(lambda f: data_dir + f, ["train.src", "dev.trg", "test.trg"]):
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


def save_vocab(datadir, vocab):
    """
    Save vocab to a dir
    """
    with open(datadir + "vocab.txt", "w+") as f:
        for word in vocab:
            f.write(word + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "actions",
        metavar="action",
        type=str,
        nargs="+",
        help="actions to take",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default=OUT_DIR,
        help="dir to write data to",
    )
    parser.add_argument(
        "--model-output",
        dest="model_prefix",
        default=MODEL_PREFIX,
        help="dir to write data to",
    )
    args = parser.parse_args()

    if "load" in args.actions:
        load_dataset(args.data_dir)

    if "train" in args.actions:
        """
        Train sentencepiece model.
        """
        sp.SentencePieceTrainer.train(
            input=args.data_dir + "/all",
            model_prefix=MODEL_PREFIX,
            vocab_size=24000,
        )
