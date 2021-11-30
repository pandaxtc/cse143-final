"""
Some basic n-gram language modeling on my own messages
"""

from nltk.lm.models import StupidBackoff
from nltk.util import bigrams, trigrams, everygrams
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_sequence, pad_both_ends
from nltk.lm import MLE, Laplace
import nltk
import json
from itertools import chain, islice


DATASET_PATH = "data/self_tokenized.json"

def asym_pad_ngram_pipeline(n, l):

    def pad(s):
        s = ["<s>"] + s
        return pad_sequence(s, n, pad_right=True, right_pad_symbol="</s>")

    return (
        (everygrams(list(pad(sent)), max_len=n, min_len=2) for sent in l),
        chain.from_iterable(map(pad, l)),
    )

if __name__ == "__main__":
    l = json.load(open(DATASET_PATH, encoding="utf8"))

    for n in range(2, 4):

        lm = MLE(0.4, n)

        # train, vocab = asym_pad_ngram_pipeline(n, l)
        # for i in range(5):
        #     t = next(train)
        #     print([g for g in t])

        train, vocab = asym_pad_ngram_pipeline(n, l)

        lm.fit(train, vocab)

        print(f"\n{n}-gram")
        for _ in range(20):
            s = ["<s>"]
            while len(s) == 0 or s[-1] != "</s>":
                s.append(lm.generate(1, text_seed=s))
                if len(s) > 50:
                    break
            print(" ".join(s))
