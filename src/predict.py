import argparse
import sys
from unittest.mock import patch

import sentencepiece as spm
from joeynmt.__main__ import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        type=str,
    )
    parser.add_argument(
        "-S",
        "--spm_path",
        dest="spm_path",
        type=str,
    )
    parser.add_argument(
        "-C",
        "--ckpt",
        dest="ckpt",
        type=str,
    )
    parser.add_argument(
        "-N",
        "--n",
        dest="n",
        type=str,
    )
    args = parser.parse_args()

    nmt_args = ["joeynmt", "translate", args.config_path]
    if args.ckpt:
        nmt_args.extend(["--ckpt", args.ckpt])
    if args.n:
        nmt_args.extend(["--n", args.n])

    preprocess = lambda l: l
    if args.spm_path:
        sp = spm.SentencePieceProcessor(model_file=args.spm_path)
        preprocess = lambda l: sp.encode(l, out_type=str)

    unpatched_input = input
    def patched_input(prompt):
        l = preprocess(unpatched_input(prompt))
        if args.spm_path:
            print(f"Preprocessed to {l}")
        return " ".join(l)

    with patch.object(
        sys, "argv", nmt_args
    ), patch("builtins.input", patched_input):
        main()
