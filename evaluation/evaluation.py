from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from jiwer import wer

smooth = SmoothingFunction()
rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
bert_scorer = BERTScorer(
    model_type="roberta-base",
    lang="en",
)


def calc_bleu(references: str, predictions: str, weights: List) -> int | float:
    ref = [references.split()]
    pred = predictions.split()
    score = sentence_bleu(ref, pred, weights=weights, smoothing_function=smooth.method1)
    return score


# I do not anaylize the case with multiple references since out dataset contains one reference for each reading task
def calc_rougeL(
    references: str, predictions: str
) -> tuple[int | float, int | float, int | float]:
    precision, recall, fmeasure = rouge_scorer.score(references, predictions)["rougeL"]
    return (precision, recall, fmeasure)


def calc_bert(
    references: str, predictions: str
) -> tuple[int | float, int | float, int | float]:
    ref = references.split()
    pred = predictions.split()
    precision, recall, f1 = bert_scorer.score(pred, ref)
    return (precision.mean().item(), recall.mean().item(), f1.mean().item())


def calc_wer(references: str, predictions: str):
    return wer(references, predictions)


def calc_accuracy(references: str, predictions: str) -> int | float:
    ref = references.split()
    pred = predictions.split()
    match = sum(r == p for r, p in zip(ref, pred))
    return match / len(ref)


def evalute(ref: str, pred: str, weights=[0.25, 0.25, 0.25, 0.25]) -> dict:
    metrics = [
        "bleu",
        "rougeL",
        "bert",
        "wer",
        "accuracy",
    ]
    eval = dict.fromkeys(metrics)
    eval["bleu"] = calc_bleu(ref, pred, weights)
    rlprecision, rlrecall, rlfmeasure = calc_rougeL(ref, pred)
    eval["rougeL"] = {"precision": rlprecision, "recall": rlrecall, "f1": rlfmeasure}
    bprecision, brecall, bf1 = calc_bert(ref, pred)
    eval["bert"] = {"precision": bprecision, "recall": brecall, "f1": bf1}
    eval["wer"] = calc_wer(ref, pred)
    eval["accuracy"] = calc_accuracy(ref, pred)
    return eval
