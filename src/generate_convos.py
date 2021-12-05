"""
The dataset is composed of processed movie scripts obtained from
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
"""

from dataclasses import dataclass, field, asdict

import json
import csv
import ast

CONVOS_PATH = "data/movies/movie_conversations.tsv"
LINES_PATH = "data/movies/movie_lines.tsv"


@dataclass
class Message:
    speaker: str
    content: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    movie: str = ""
    participants: list[str] = field(default_factory=set)
    messages: list[Message] = field(default_factory=list)


if __name__ == "__main__":
    convos: list[Conversation] = []
    line_map: dict[str, tuple[str, str, str]] = {}

    with open(LINES_PATH, encoding="utf8") as f:
        c = csv.reader(
            f, delimiter="\t", lineterminator="\n", strict=True, quoting=csv.QUOTE_NONE
        )
        next(c) # skip header

        for n, entry in enumerate(c):
            try:
                line, speaker, movie, speaker_name, speech = entry
                assert line not in line_map
                line_map[line] = (speaker, movie, speech)
            except ValueError as e:
                print(n, e, entry)

    with open(CONVOS_PATH, encoding="utf8") as f:
        c = csv.reader(
            f, delimiter="\t", lineterminator="\n", strict=True, quoting=csv.QUOTE_NONE
        )
        next(c)

        for entry in c:
            speaker_1, speaker_2, movie, lines = entry

            try:
                lines = ast.literal_eval(lines)
            except ValueError as e:
                print(entry)

            convo = Conversation(participants=[speaker_1, speaker_2], movie=movie)

            prev_speaker = None

            for line in lines:
                speaker, _movie, speech = line_map[line]
                if speech.strip() == "":
                    continue

                assert speaker in convo.participants
                assert movie == _movie

                if speaker == prev_speaker:
                    convo.messages[-1].content.append(speech)
                else:
                    convo.messages.append(Message(speaker, [speech]))

            convos.append(convo)

    json.dump(
        [asdict(convo) for convo in convos],
        open("data/movies.json", "w", encoding="utf8"),
        indent="\t",
        ensure_ascii=False,
    )
