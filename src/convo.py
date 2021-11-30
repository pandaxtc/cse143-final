"""
The dataset is composed of exports from Google Hangouts and Facebook
Messenger. 

I'm choosing to exclude group chats and only look at DMs for now,
because conversations in group chats are too hard to keep track of.

The Hangouts data is a single extremely large JSON file. The messenger
data is a series of directories representing groups/DMs with messages in
a JSON file.

For some reason, multibyte UTF-8 sequences are represented incorrectly
in some datasets. I use ftfy to fix this. Thanks ftfy devs!
"""

from dataclasses import dataclass, field, asdict
from glob import glob

import json
import ftfy
import unicodedata
import numpy

HANGOUTS_PATH = "data/messages/google/Hangouts/Hangouts.json"
MESSENGER_PATH = "data/messages/facebook/*/*.json"
ME = "Waylon Peng"
HANGOUTS_ID = "102894839669481941064"


@dataclass
class Message:
    sender_id: str
    timestamp: str
    content: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    participants: dict[str, str] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)


"""
Sadly because the dataset is derived from my messages as a kid, it
contains a lot of Zalgo text. Here's some methods to try and fix it.
"""
ZALGO_CHAR_CATEGORIES = ["Mn", "Me"]
THRESHOLD = 0.5


def is_zalgo(s):
    if len(s) == 0:
        return False
    word_scores = []
    for word in s.split():
        cats = [unicodedata.category(c) for c in word]
        score = sum([cats.count(banned) for banned in ZALGO_CHAR_CATEGORIES]) / len(
            word
        )
        word_scores.append(score)

    if not word_scores:
        return False

    total_score = numpy.percentile(word_scores, 75)
    return total_score > THRESHOLD


def clean(s):
    if is_zalgo(s):
        s = "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) not in ZALGO_CHAR_CATEGORIES
        )
    return ftfy.fix_text(s).strip()


if __name__ == "__main__":
    convos = []

    """
    Load data from Hangouts
    """
    h = json.load(open(HANGOUTS_PATH, encoding="utf8"))
    for c in h["conversations"]:
        if (
            c["conversation"]["conversation"]["type"] != "STICKY_ONE_TO_ONE"
            or len(c["events"]) < 2
        ):
            continue

        convo = Conversation()
        for p in c["conversation"]["conversation"]["participant_data"]:
            if p["id"]["chat_id"] == HANGOUTS_ID:
                convo.participants[p["id"]["chat_id"]] = ME
            elif "fallback_name" in p:
                convo.participants[p["id"]["chat_id"]] = p["fallback_name"]
            else:
                convo.participants[p["id"]["chat_id"]] = "unknown"

        print(f"processing hangouts convo {convo.participants}")

        msgs = [
            msg
            for msg in c["events"]
            if msg["event_type"] == "REGULAR_CHAT_MESSAGE"
            and "segment" in msg["chat_message"]["message_content"]
            and any(
                "text" in seg
                for seg in msg["chat_message"]["message_content"]["segment"]
            )
        ]

        if len(msgs) < 2:
            print("!! no chat messages in convo")
            continue

        msgs.sort(key=lambda m: m["timestamp"])

        last_id = None

        for msg in msgs:
            if msg["sender_id"]["chat_id"] == last_id:
                convo.messages[-1].content.append(
                    clean(
                        " ".join(
                            map(
                                lambda seg: seg["text"] if "text" in seg else "",
                                msg["chat_message"]["message_content"]["segment"],
                            )
                        )
                    )
                )
            else:
                convo.messages.append(
                    Message(
                        convo.participants.get(
                            msg["sender_id"]["chat_id"], msg["sender_id"]["chat_id"]
                        ),
                        msg["timestamp"],
                        [
                            clean(
                                " ".join(
                                    map(
                                        lambda seg: seg["text"]
                                        if "text" in seg
                                        else "",
                                        msg["chat_message"]["message_content"][
                                            "segment"
                                        ],
                                    )
                                )
                            )
                        ],
                    )
                )
                last_id = msg["sender_id"]["chat_id"]

        convos.append(convo)

    del h

    """
    Load data from Messenger
    """
    for f in glob(MESSENGER_PATH):
        with open(f, encoding="utf8") as f:
            c = json.load(f)
            if c["thread_type"] != "Regular":
                continue

            convo = Conversation()
            for p in c["participants"]:
                convo.participants[p["name"]] = p["name"]

            print(f"processing messenger convo {convo.participants}")

            msgs = [
                msg
                for msg in c["messages"]
                if "content" in msg
                and not msg["content"].lower().startswith("words with friends:")
                and msg["content"] != ""
            ]

            if len(msgs) < 2:
                print("!! no chat messages in convo")
                continue

            msgs.sort(key=lambda m: m["timestamp_ms"])

            last_id = None

            for msg in msgs:
                if msg["sender_name"] == last_id:
                    convo.messages[-1].content.append(clean(msg["content"]))
                else:
                    convo.messages.append(
                        Message(
                            msg["sender_name"],
                            str(msg["timestamp_ms"]),
                            [clean(msg["content"])],
                        )
                    )
                    last_id = msg["sender_name"]

            convos.append(convo)

    json.dump(
        [asdict(convo) for convo in convos],
        open("data/messages.json", "w", encoding="utf8"),
        indent="\t",
        ensure_ascii=False,
    )
