import argparse
import subprocess
import sys

import sentencepiece as spm

"""
######################################
The following is adapted from JoeyNMT.
######################################
"""
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from joeynmt.batch import Batch
from joeynmt.constants import EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from joeynmt.data import MonoDataset, load_data, make_data_iter
from joeynmt.helpers import (
    bpe_postprocess,
    expand_reverse_index,
    get_latest_checkpoint,
    load_checkpoint,
    load_config,
    make_logger,
    store_attention_plots,
)
from joeynmt.metrics import bleu, chrf, sequence_accuracy, token_accuracy
from joeynmt.model import Model, _DataParallel, build_model
from joeynmt.search import run_batch
from joeynmt.vocabulary import Vocabulary
from torchtext.legacy.data import Dataset, Field  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)


def parse_test_args(cfg, mode="test"):
    """
    parse test args
    :param cfg: config object
    :param mode: 'test' or 'translate'
    :return:
    """
    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    batch_size = cfg["training"].get(
        "eval_batch_size", cfg["training"].get("batch_size", 1)
    )
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence")
    )
    use_cuda = cfg["training"].get("use_cuda", False) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if mode == "test":
        n_gpu = torch.cuda.device_count() if use_cuda else 0
        # k = cfg["testing"].get("beam_size", 1)
        # batch_per_device = batch_size*k // n_gpu if n_gpu > 1 else batch_size*k
        batch_per_device = batch_size // n_gpu if n_gpu > 1 else batch_size
        logger.info(
            "Process device: %s, n_gpu: %d, batch_size per device: %d",
            device,
            n_gpu,
            batch_per_device,
        )
        eval_metric = cfg["training"]["eval_metric"]

    elif mode == "translate":
        # in multi-gpu, batch_size must be bigger than n_gpu!
        n_gpu = 1 if use_cuda else 0
        logger.debug("Process device: %s, n_gpu: %d", device, n_gpu)
        eval_metric = ""

    level = cfg["data"]["level"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 1)
        beam_alpha = cfg["testing"].get("alpha", -1)
        postprocess = cfg["testing"].get("postprocess", True)
        bpe_type = cfg["testing"].get("bpe_type", "subword-nmt")
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
        if "sacrebleu" in cfg["testing"].keys():
            sacrebleu["remove_whitespace"] = cfg["testing"]["sacrebleu"].get(
                "remove_whitespace", True
            )
            sacrebleu["tokenize"] = cfg["testing"]["sacrebleu"].get("tokenize", "13a")

    else:
        beam_size = 1
        beam_alpha = -1
        postprocess = True
        bpe_type = "subword-nmt"
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}

    decoding_description = (
        "Greedy decoding"
        if beam_size < 2
        else "Beam search decoding with beam size = {} and alpha = {}".format(
            beam_size, beam_alpha
        )
    )
    tokenizer_info = f"[{sacrebleu['tokenize']}]" if eval_metric == "bleu" else ""

    return (
        batch_size,
        batch_type,
        use_cuda,
        device,
        n_gpu,
        level,
        eval_metric,
        max_output_length,
        beam_size,
        beam_alpha,
        postprocess,
        bpe_type,
        sacrebleu,
        decoding_description,
        tokenizer_info,
    )


# pylint: disable=too-many-arguments,too-many-locals,no-member,too-many-branches
def validate_on_data(
    model: Model,
    data: Dataset,
    batch_size: int,
    use_cuda: bool,
    max_output_length: int,
    level: str,
    eval_metric: Optional[str],
    n_gpu: int,
    batch_class: Batch = Batch,
    compute_loss: bool = False,
    beam_size: int = 1,
    beam_alpha: int = -1,
    batch_type: str = "sentence",
    postprocess: bool = True,
    bpe_type: str = "subword-nmt",
    sacrebleu: dict = None,
    n_best: int = 1,
):
    """
    Generate translations for the given data.
    If `compute_loss` is True and references are given,
    also compute the loss.
    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param batch_class: class type of batch
    :param use_cuda: if True, use CUDA
    :param max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param eval_metric: evaluation metric, e.g. "bleu"
    :param n_gpu: number of GPUs
    :param compute_loss: whether to computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation.
        If <2 then greedy decoding (default).
    :param beam_alpha: beam search alpha for length penalty,
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param postprocess: if True, remove BPE segmentation from translations
    :param bpe_type: bpe type, one of {"subword-nmt", "sentencepiece"}
    :param sacrebleu: sacrebleu options
    :param n_best: Amount of candidates to return
    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    assert batch_size >= n_gpu, "batch_size must be bigger than n_gpu."
    if sacrebleu is None:  # assign default value
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like "
            "this? 'batch_size' is > 1000 for sentence-batching. "
            "Consider decreasing it or switching to"
            " 'eval_batch_type: token'."
        )
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        batch_type=batch_type,
        shuffle=False,
        train=False,
    )
    valid_sources_raw = data.src
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = []
        total_loss = 0
        total_ntokens = 0
        total_nseqs = 0
        for valid_batch in iter(valid_iter):
            # run as during training to get validation loss (e.g. xent)

            batch = batch_class(valid_batch, pad_index, use_cuda=use_cuda)
            if batch.nseqs < 1:
                continue

            # sort batch now by src length and keep track of order
            reverse_index = batch.sort_by_src_length()
            sort_reverse_index = expand_reverse_index(reverse_index, n_best)

            # run as during training with teacher forcing
            if compute_loss and batch.trg is not None:
                batch_loss, _, _, _ = model(return_type="loss", **vars(batch))
                if n_gpu > 1:
                    batch_loss = batch_loss.mean()  # average on multi-gpu
                total_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # run as during inference to produce translations
            output, attention_scores = run_batch(
                model=model,
                batch=batch,
                beam_size=beam_size,
                beam_alpha=beam_alpha,
                max_output_length=max_output_length,
                n_best=n_best,
            )

            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])
            valid_attention_scores.extend(
                attention_scores[sort_reverse_index]
                if attention_scores is not None
                else []
            )

        assert len(all_outputs) == len(data) * n_best

        if compute_loss and total_ntokens > 0:
            # total validation loss
            valid_loss = total_loss
            # exponent of token-level negative log prob
            valid_ppl = torch.exp(total_loss / total_ntokens)
        else:
            valid_loss = -1
            valid_ppl = -1

        # decode back to symbols
        decoded_valid = model.trg_vocab.arrays_to_sentences(
            arrays=all_outputs, cut_at_eos=True
        )

        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        valid_sources = [join_char.join(s) for s in data.src]
        valid_references = [join_char.join(t) for t in data.trg]
        valid_hypotheses = [join_char.join(t) for t in decoded_valid]

        # post-process
        if level == "bpe" and postprocess:
            valid_sources = [
                bpe_postprocess(s, bpe_type=bpe_type) for s in valid_sources
            ]
            valid_references = [
                bpe_postprocess(v, bpe_type=bpe_type) for v in valid_references
            ]
            valid_hypotheses = [
                bpe_postprocess(v, bpe_type=bpe_type) for v in valid_hypotheses
            ]

        # if references are given, evaluate against them
        if valid_references:
            assert len(valid_hypotheses) == len(valid_references)

            current_valid_score = 0
            if eval_metric.lower() == "bleu":
                # this version does not use any tokenization
                current_valid_score = bleu(
                    valid_hypotheses, valid_references, tokenize=sacrebleu["tokenize"]
                )
            elif eval_metric.lower() == "chrf":
                current_valid_score = chrf(
                    valid_hypotheses,
                    valid_references,
                    remove_whitespace=sacrebleu["remove_whitespace"],
                )
            elif eval_metric.lower() == "token_accuracy":
                current_valid_score = token_accuracy(  # supply List[List[str]]
                    list(decoded_valid), list(data.trg)
                )
            elif eval_metric.lower() == "sequence_accuracy":
                current_valid_score = sequence_accuracy(
                    valid_hypotheses, valid_references
                )
        else:
            current_valid_score = -1

    return (
        current_valid_score,
        valid_loss,
        valid_ppl,
        valid_sources,
        valid_sources_raw,
        valid_references,
        valid_hypotheses,
        decoded_valid,
        valid_attention_scores,
    )


def translate(
    cfg_file: str, ckpt: str, batch_class: Batch = Batch, n_best: int = 1
) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or
    asks for input to translate interactively.
    The input has to be pre-processed according to the data that the model
    was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output file
    :param batch_class: class type of batch
    :param n_best: amount of candidates to display
    """

    def _load_line_as_data(line):
        """Create a dataset from one line via a temporary file."""
        # write src input to temporary file
        tmp_name = "tmp"
        tmp_suffix = ".src"
        tmp_filename = tmp_name + tmp_suffix
        with open(tmp_filename, "w", encoding="utf-8") as tmp_file:
            tmp_file.write("{}\n".format(line))

        test_data = MonoDataset(path=tmp_name, ext=tmp_suffix, field=src_field)

        # remove temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        return test_data

    def _translate_data(test_data):
        """Translates given dataset, using parameters from outer scope."""
        # pylint: disable=unused-variable
        (
            score,
            loss,
            ppl,
            sources,
            sources_raw,
            references,
            hypotheses,
            hypotheses_raw,
            attention_scores,
        ) = validate_on_data(
            model,
            data=test_data,
            batch_size=batch_size,
            batch_class=batch_class,
            batch_type=batch_type,
            level=level,
            max_output_length=max_output_length,
            eval_metric="",
            use_cuda=use_cuda,
            compute_loss=False,
            beam_size=beam_size,
            beam_alpha=beam_alpha,
            postprocess=postprocess,
            bpe_type=bpe_type,
            sacrebleu=sacrebleu,
            n_gpu=n_gpu,
            n_best=n_best,
        )
        return hypotheses

    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]

    _ = make_logger(model_dir, mode="translate")
    # version string returned

    # when checkpoint is not specified, take oldest from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir)

    # read vocabs
    src_vocab_file = cfg["data"].get("src_vocab", model_dir + "/src_vocab.txt")
    trg_vocab_file = cfg["data"].get("trg_vocab", model_dir + "/trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)

    data_cfg = cfg["data"]
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = Field(
        init_token=None,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        tokenize=tok_fun,
        batch_first=True,
        lower=lowercase,
        unk_token=UNK_TOKEN,
        include_lengths=True,
    )
    src_field.vocab = src_vocab

    # parse test args
    (
        batch_size,
        batch_type,
        use_cuda,
        device,
        n_gpu,
        level,
        _,
        max_output_length,
        beam_size,
        beam_alpha,
        postprocess,
        bpe_type,
        sacrebleu,
        _,
        _,
    ) = parse_test_args(cfg, mode="translate")

    # load model state from disk
    logger.info("Loading model from %s", ckpt)
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.to(device)

    batch_size = 1
    batch_type = "sentence"
    while True:
        try:
            src_input = input("\nPlease enter a source sentence " "(pre-processed): \n")
            if not src_input.strip():
                break

            # every line has to be made into dataset
            test_data = _load_line_as_data(line=src_input)
            hypotheses = _translate_data(test_data)

            print("JoeyNMT: Hypotheses ranked by score")
            for i, hyp in enumerate(hypotheses):
                print("JoeyNMT #{}: {}".format(i + 1, hyp))

        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break


"""
#################
End JoeyNMT code.
#################
"""


preprocess = lambda l: l


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
        "--best_n",
        dest="best_n",
        type=int,
    )
    args = parser.parse_args()

    nmt_args = [sys.executable, "-m", "joeynmt", "translate", args.config_path]
    if args.ckpt:
        nmt_args.extend(["--ckpt", args.ckpt])
    if args.best_n:
        nmt_args.extend(["--best_n", args.best_n])
    if args.spm_path:
        sp = spm.SentencePieceProcessor(model_file=args.spm_path)
        preprocess = lambda l: sp.encode(l)

    translate()
