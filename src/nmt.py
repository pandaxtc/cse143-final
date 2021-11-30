"""
Marshal raw messages into data suitable for NMT training.

JoeyNMT takes sentence, translation pairs. This is a little tricky,
because sentence boundaries are really wishy-washy in chats, and a
sentence can be a response to many senteces, so it's not 1:1. Oh well!
"""

from numpy.lib.npyio import save
from convo import ME, Conversation
from util import load_convos
from nltk import TweetTokenizer, tokenize

import numpy as np
import os

OUT_DIR = "data/joeynmt/messages/"

tokenizer = TweetTokenizer(reduce_len=True)


def clean(s: str):
    return s.lower().replace("\n", " ")


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
                " ".join(tokenizer.tokenize(" ".join(map(clean, message.content))))
            )
        else:
            theirs.append(
                " ".join(tokenizer.tokenize(" ".join(map(clean, message.content))))
            )

    if len(theirs) > len(mine):
        assert len(theirs) == len(mine) + 1
        theirs.pop()

    assert len(mine) == len(theirs)
    assert not any(a == "" or b == "" for a, b in zip(mine, theirs))
    return mine, theirs


def save_tl(datadir, source, target, prefix):
    with open(datadir + prefix + ".src", "w+") as f:
        for line in source:
            f.write(line + "\n")
    with open(datadir + prefix + ".trg", "w+") as f:
        for line in target:
            f.write(line + "\n")


if __name__ == "__main__":
    mine, theirs = [], []

    for convo in load_convos():
        assert ME in convo.participants.values(), str(convo)

        convo_mine, convo_theirs = naive_tokenize(convo)
        mine.extend(convo_mine)
        theirs.extend(convo_theirs)

    assert len(mine) == len(theirs)

    mine, theirs = map(np.array, (mine, theirs))

    rng = np.random.default_rng(420)
    indices = rng.permutation(mine.shape[0])

    train, dev, test = np.split(indices, [int(0.6 * len(mine)), int(0.8 * len(mine))])

    train_trg, train_src = mine[train], theirs[train]
    dev_trg, dev_src = mine[dev], theirs[dev]
    test_trg, test_src = mine[test], theirs[test]

    save_tl(OUT_DIR, train_src, train_trg, "train")
    save_tl(OUT_DIR, dev_src, dev_trg, "dev")
    save_tl(OUT_DIR, test_src, test_trg, "test")
