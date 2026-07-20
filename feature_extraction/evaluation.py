import torch
import torchmetrics
from torchmetrics import text
from datasets import load_me

accuracy = load_metric("accuracy")
bleu = load_metric("bleu")
# n-gram is basically n words; "4 word matches/4 unigram matches"
# n > 1 provides a way to account for the order of tokens in a sentence
rouge = load_metric("rouge")
bert = load_metric("bertscore")
wer = load_metric("wer")

reference = ["I", "love", "Han", "Jisung"]
predicted = ["I", "Jisung", "Han", "hate"]

bleu.compute(predictions=predicted, refereces=reference)