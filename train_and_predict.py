import os
import pickle
import argparse
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from dnn_model import DNNModel
from config import *


def load_train_data():
    with open(TRAIN_SEQUENCES_BINARY, "rb") as f:
        learn_sequences = pickle.load(f)
    with open(TRAIN_LABELS_BINARY, "rb") as f:
        learn_labels = pickle.load(f)
    return learn_sequences, learn_labels


def load_test_data():
    with open(TEST_SEQUENCES_BINARY, "rb") as f:
        test_sequences = pickle.load(f)
    return test_sequences


def save_predictions(predictions, name):
    path = os.path.join(PREDICTED_LABELS_FOLDER, "labels_" + name + ".pkl")
    with open(path, "wb") as f:
        pickle.dump(list(predictions), f)
    return path


def tokens_to_str(tokens):
    return list(map(str, tokens))


def get_weight_for_class_M(y):
    counts = Counter(y)
    return counts["C"] / counts["M"]


def create_model(model_type, weight_M):
    if model_type == "naive_bayes":
        model = Pipeline([
            ("vectorizer", CountVectorizer(
                lowercase=False, tokenizer=tokens_to_str,ngram_range=(1, 2))),
            ("classifier", MultinomialNB(alpha=0.1))
        ])
    elif model_type == "dnn":
        print("Using weight = %.2f for class 'M'" % weight_M)
        model = DNNModel(weight_class_M=weight_M)
    else:
        raise ValueError("Incorrect model_type value")
        
    return model


def train_and_predict(model_type):
    X_train, y_train = load_train_data()
    print("Loaded train data")

    # Weight for class M (only used in dnn model)
    weight_M = get_weight_for_class_M(y_train)   
    
    model = create_model(model_type, weight_M)
    model.fit(X_train, y_train)
    print("Trained model")

    X_test = load_test_data()
    print("Loaded test data")

    predictions = model.predict(X_test)
    save_path = save_predictions(predictions, model_type)
    print("Saved predictions to %s" % save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", help="choose which type of model to use",
        choices=["naive_bayes", "dnn"], required=True
    )
    args = parser.parse_args()
    print("\nUsing '%s' model" % args.model_type)
    train_and_predict(args.model_type)
