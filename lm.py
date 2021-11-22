"""
Some basic n-gram language modeling on my own messages
"""

from nltk.util import bigrams, trigrams, everygrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import nltk
import json
import re







if __name__ == "__main__":
    l = load_dataset("data/messages.json")

    bigram_lm = MLE(2)
    bigram_training, vocab = padded_everygram_pipeline(2, l)
    bigram_lm.fit(bigram_training, vocab)

    trigram_lm = MLE(3)
    trigram_training, vocab = padded_everygram_pipeline(3, l)
    trigram_lm.fit(trigram_training, vocab)

    print("\nbigram")
    for _ in range(5):
        s = ["<s>"]
        while len(s) == 0 or s[-1] != "</s>":
            s.append(bigram_lm.generate(1, text_seed=s))
            if len(s) > 50:
                break
        print(" ".join(s))

    print("\ntrigram")
    for _ in range(5):
        s = ["<s>"]
        while len(s) == 0 or s[-1] != "</s>":
            s.append(trigram_lm.generate(1, text_seed=s))
            if len(s) > 50:
                break
        print(" ".join(s))
