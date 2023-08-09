from typing import Union

from fastapi import FastAPI
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score
import numpy as np
from small_text import TransformersDataset
from transformers import AutoTokenizer
from enum import Enum

from small_text import (
    EmptyPoolException,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    RandomSampling,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
    random_initialization_balanced
)

app = FastAPI()

class HFModel(str, Enum):
    """List all HF models available for classification"""
    distillroberta = "distilroberta-base"
    roberta = "roberta-base"
    longformer = "allenai/longformer-base-4096"
    clinicallongformer = "yikuan8/Clinical-Longformer"


TWENTY_NEWS_SUBCATEGORIES = ['rec.sport.baseball', 'sci.med', 'rec.autos']


def get_twenty_newsgroups_corpus(categories=TWENTY_NEWS_SUBCATEGORIES):

    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                               categories=categories)

    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                              categories=categories)

    return train, test


def preprocess_data(tokenizer, texts, labels, max_length=500):
    return TransformersDataset.from_arrays(texts, labels, tokenizer, max_length=max_length)


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    print('Train accuracy: {:.2f}'.format(
        f1_score(y_pred, train.y, average='micro')))
    print('Test accuracy: {:.2f}'.format(f1_score(y_pred_test, test.y, average='micro')))
    print('---')


@app.get("/")
def read_root():
    return {"Hello": "World"}
