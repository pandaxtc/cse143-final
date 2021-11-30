from convo import Conversation, Message

import json

DATA_DIR = "data/messages.json"

def load_convos(file=DATA_DIR):
    with open(file, encoding="utf8") as f:
        convos = json.load(f)
        for convo in map(dict_to_convo, convos):
            yield convo


def dict_to_convo(d):
    return Conversation(d["participants"], [Message(**msg) for msg in d["messages"]])
