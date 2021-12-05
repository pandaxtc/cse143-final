from generate_convos import Conversation, Message

import json

DATA_DIR = "data/movies.json"

def load_convos(file=DATA_DIR):
    with open(file, encoding="utf8") as f:
        convos = json.load(f)
        for convo in map(dict_to_convo, convos):
            if len(convo.messages) >= 2:
                yield convo

def dict_to_convo(d):
    return Conversation(participants=d["participants"], messages=[Message(**msg) for msg in d["messages"]])
