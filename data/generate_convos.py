"""
The dataset is composed of exports from Google Hangouts and Facebook
Messenger. 

I'm choosing to exclude group chats and only look at DMs for now,
because conversations in group chats are too hard to keep track of.

The Hangouts data is a single extremely large JSON file. The messenger
data is a series of directories representing groups/DMs with messages in
a JSON file.
"""

from dataclasses import dataclass, field, asdict
from glob import glob

import json

HANGOUTS_PATH = "messages/google/Hangouts/Hangouts.json"
MESSENGER_PATH = "messages/facebook/*/*.json"
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


if __name__ == "__main__":
    convos = []

    """
    Load data from Hangouts
    """
    h = json.load(open(HANGOUTS_PATH))
    for c in h["conversations"]:
        if (
            c["conversation"]["conversation"]["type"] != "STICKY_ONE_TO_ONE"
            or len(c["events"]) < 2
        ):
            continue

        convo = Conversation()
        for p in c["conversation"]["conversation"]["participant_data"]:
            if "fallback_name" in p:
                convo.participants[p["id"]["chat_id"]] = p["fallback_name"]
            elif p["id"]["chat_id"] == HANGOUTS_ID:
                convo.participants[p["id"]["chat_id"]] = ME
            else:
                convo.participants[p["id"]["chat_id"]] = "unknown"

        print(f"processing hangouts convo {convo.participants}")

        msgs = [
            msg
            for msg in c["events"]
            if msg["event_type"] == "REGULAR_CHAT_MESSAGE"
            and "segment" in msg["chat_message"]["message_content"]
        ]

        if len(msgs) < 2:
            print("!! no chat messages in convo")
            continue

        msgs.sort(key=lambda m: m["timestamp"])

        last_id = None

        for msg in msgs:
            if msg["sender_id"]["chat_id"] == last_id:
                convo.messages[-1].content.append(
                    msg["chat_message"]["message_content"]["segment"][0]["text"]
                )
            else:
                convo.messages.append(
                    Message(
                        convo.participants.get(
                            msg["sender_id"]["chat_id"], msg["sender_id"]["chat_id"]
                        ),
                        msg["timestamp"],
                        [msg["chat_message"]["message_content"]["segment"][0]["text"]],
                    )
                )
                last_id = msg["sender_id"]["chat_id"]

        convos.append(convo)

    del h

    """
    Load data from Messenger
    """
    for f in glob(MESSENGER_PATH):
        with open(f) as f:
            c = json.load(f)
            if c["thread_type"] != "Regular":
                continue

            convo = Conversation()
            for p in c["participants"]:
                convo.participants[p["name"]] = p["name"]

            print(f"processing messenger convo {convo.participants}")

            msgs = [msg for msg in c["messages"] if "content" in msg]
            msgs.sort(key=lambda m: m["timestamp_ms"])

            last_id = None

            for msg in msgs:
                if msg["sender_name"] == last_id:
                    convo.messages[-1].content.append(msg["content"])
                else:
                    convo.messages.append(
                        Message(
                            msg["sender_name"],
                            str(msg["timestamp_ms"]),
                            [msg["content"]],
                        )
                    )
                    last_id = msg["sender_name"]

            convos.append(convo)

    json.dump([asdict(convo) for convo in convos], open("messages.json", "w"), indent="\t")
