import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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


def save_predictions(predictions):
    with open(PREDICTED_LABELS_BINARY, "wb") as f:
        pickle.dump(predictions, f)


def tokens_to_str(tokens):
    return list(map(str, tokens))


def create_model():
    model = Pipeline([
        ("vectorizer", CountVectorizer(lowercase=False, tokenizer=tokens_to_str,
                                       ngram_range=(1, 2))),
        ("classifier", MultinomialNB(alpha=0.1))
    ])
    return model


def train_and_predict():
    X_train, y_train = load_train_data()
    print("Loaded train data")

    model = create_model()
    model.fit(X_train, y_train)
    print("Trained model")

    X_test = load_test_data()
    print("Loaded test data")

    predictions = model.predict(X_test)
    save_predictions(predictions)
    print("Saved predictions to %s" % PREDICTED_LABELS_BINARY)
        

if __name__ == "__main__":
    train_and_predict()

    