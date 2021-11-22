from data.generate_convos import ME, Message, Conversation
#from nnsplit import NNSplit

def load_dataset(file, n=2):
    sents = []
    
    # tk = nltk.TweetTokenizer(preserve_case=False, reduce_len=True)
    tk = NNSplit("en")
    
    with open(file) as f:
        convos = json.load(f)
        for convo in map(dict_to_convo, convos):
            assert "Waylon Peng" in convo.participants.values(), str(convo)
            for message in convo.messages:
                if message.sender_id == ME:
                    sent = " ".join(message.content)
                    sents.append(tk.split(sent))
    return sents

def dict_to_convo(d):
    return Conversation(d["participants"], [Message(**msg) for msg in d["messages"]])