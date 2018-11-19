import os
import pickle
import argparse
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from dnn_model import DNNModel
from cnn_model import CNNModel
from config import *


def load_train_data(custom=False, shuffle=False):
    binary = CUSTOM_TRAIN_SEQUENCES if custom else TRAIN_SEQUENCES
    with open(binary, "rb") as f:
        learn_sequences = np.array(pickle.load(f))
    with open(TRAIN_LABELS, "rb") as f:
        learn_labels = np.array(pickle.load(f))

    if shuffle:
        learn_sequences, learn_labels = shuffle_data(learn_sequences, learn_labels)
    return learn_sequences, learn_labels


def load_test_data(custom=False):
    binary = CUSTOM_TEST_SEQUENCES if custom else TEST_SEQUENCES
    with open(binary, "rb") as f:
        test_sequences = pickle.load(f)
    return test_sequences


def load_embeddings():
    with open(CUSTOM_EMBEDDINGS, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings    


def save_predictions(predictions, name):
    path = os.path.join(PREDICTED_LABELS_FOLDER, "labels_" + name + ".pkl")
    with open(path, "wb") as f:
        pickle.dump(list(predictions), f)
    return path


def shuffle_data(X, y):
    permutation = np.random.permutation(len(y))
    X, y = X[permutation], y[permutation]
    return X, y


def tokens_to_str(tokens):
    return list(map(str, tokens))


def get_weight_for_class_M(y):
    counts = Counter(y)
    return counts["C"] / counts["M"]


def create_model(model_type, X_train, y_train):
    if model_type == "naive_bayes":
        model = Pipeline([
            ("vectorizer", CountVectorizer(
                lowercase=False, tokenizer=tokens_to_str,ngram_range=(1, 2))),
            ("classifier", MultinomialNB(alpha=0.1))
        ])
        
    elif model_type == "dnn":
        weight_M = get_weight_for_class_M(y_train)   
        model = DNNModel(weight_class_M=weight_M)
        
    elif model_type == "cnn":
        weight_M = get_weight_for_class_M(y_train)
        sentence_length = max(map(len, X_train))
        embeddings = load_embeddings()
        model = CNNModel(
            sentence_length=sentence_length,
            embeddings=embeddings,
            weight_class_M=weight_M, 
        )
        
    else:
        raise ValueError("Incorrect model_type value")
        
    return model


def train_and_predict(model_type):
    custom = True if model_type in {"dnn", "cnn"} else False
    
    X_train, y_train = load_train_data(custom=custom, shuffle=True)
    print("Loaded train data")

    model = create_model(model_type, X_train, y_train)
    model.fit(X_train, y_train)
    print("Trained model")

    X_test = load_test_data(custom=custom)
    print("Loaded test data")

    predictions = model.predict(X_test)
    save_path = save_predictions(predictions, model_type)
    print("Saved predictions to %s" % save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", help="choose which type of model to use",
        choices=["naive_bayes", "dnn", "cnn"], required=True
    )
    args = parser.parse_args()
    print("\nUsing '%s' model" % args.model_type)
    train_and_predict(args.model_type)
