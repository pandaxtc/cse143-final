"""
Tokenizing my own messages for use in exploratory ngram models.

Sentence boundary identification is pretty hard in these cases. I'm
experimenting with it here.

TODO: try SpaCy?
"""
from data.generate_convos import ME, Message, Conversation
from nnsplit import NNSplit
import json
import nltk

DATA_PATH = "data/messages.json"
SELF_OUT_PATH = "data/self_tokenized.json"

tweet_tokenizer = nltk.TweetTokenizer(preserve_case=False, reduce_len=True)
nnsplit_tokenizer = NNSplit.load("en")

def tweet_tokenize(s):
    s = " ".join(s)
    return [tweet_tokenizer.tokenize(s)]

def nnsplit_tokenize(s):
    s = " ".join(s)
    splits = nnsplit_tokenizer.split([s])[0]
    return [[str(word).strip() for word in sent] for sent in splits]

def per_msg_tweet_tokenize(s):
    sents = []
    for sent in s:
        sents.append(tweet_tokenizer.tokenize(sent))
    return sents

def generate_self_tokens(file):
    sents = []

    with open(file, encoding="utf8") as f:
        convos = json.load(f)
        for convo in map(dict_to_convo, convos):
            assert ME in convo.participants.values(), str(convo)
            for message in convo.messages:
                if message.sender_id == ME:
                    sents.extend(per_msg_tweet_tokenize(message.content))
    return sents


def dict_to_convo(d):
    return Conversation(d["participants"], [Message(**msg) for msg in d["messages"]])


if __name__ == "__main__":
    json.dump(
        generate_self_tokens(DATA_PATH),
        open(SELF_OUT_PATH, "w", encoding="utf8"),
        indent="\t",
        ensure_ascii=False,
    )
