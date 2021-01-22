from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

ctxs = [mx.cpu()] # or, e.g., [mx.gpu(0), mx.gpu(1)]

sentence = 'confirms HTTPURL via @USER :cry:'

print('Checking original MLM library..')
# MXNet MLMs (use names from mlm.models.SUPPORTED_MLMS)
model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')

#print(type(vocab).__name__)
scorer = MLMScorer(model, vocab, tokenizer, ctxs)
print(scorer.score_sentences([sentence]))
# >> [-12.410664200782776]
print(scorer.score_sentences([sentence], per_token=True))
# >> [[None, -6.126736640930176, -5.501412391662598, -0.7825151681900024, None]]

print('Done. Checking extension..')
# Load the AutoTokenizer with a normalization mode if the input Tweet is raw
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

bertweet, vocab, tokenizer = get_pretrained(ctxs, 'vinai/bertweet-base-en-cased')

#print(BERTVocab(tokenizer.vocab_file))

tweetscorer = MLMScorerPT(bertweet, None, tokenizer, ctxs)

print(tweetscorer.score_sentences([sentence]))

print(tweetscorer.score_sentences([sentence], per_token=True))

print('Done. Evaluating on waseem..')

with open('../../waseem_abusive.tsv', 'r') as f:
	abusive_scores = tweetscorer.score_sentences([line.strip() for line in f.readlines()])
print('\nScore on abusive tweets: {}, {}, {}, {}\n'.format(np.mean(abusive_scores), np.std(abusive_scores), max(abusive_scores), min(abusive_scores)))
with open('../../waseem_non_abusive.tsv', 'r') as f:
	non_abusive_scores = tweetscorer.score_sentences([line.strip() for line in f.readlines()])
print('\nScore on non abusive tweets: {}, {}, {}, {}\n'.format(np.mean(non_abusive_scores), np.std(abusive_scores), max(non_abusive_scores), min(non_abusive_scores)))

print('\nDone!')