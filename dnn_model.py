import os
import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

tf.logging.set_verbosity(tf.logging.INFO)


class DNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 model_name=None,
                 checkpoints_dir="../checkpoints/",
                 hidden_units=(250,),
                 batch_size=128,
                 weight_class_M=1.0,
                 features_key="x",
                 weight_key="weight"
                 ):
        self.set_model_directory(checkpoints_dir, model_name)
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.features_key = features_key
        self.weight_key = weight_key
        self.weight_class_M = weight_class_M

    def fit(self, X, y, num_epochs=10, warm_start=False):
        warm_start = self.check_warm_start(warm_start)
        if not warm_start:
            X, y = self.fit_and_apply_transformers(X, y)
            self.classifier_ = self.create_dnn_classifier()
        else:
            X, y = self.apply_transformers(X, y)

        self.classifier_.train(self.train_input_fn(X, y, num_epochs))
        return self

    def fit_and_apply_transformers(self, X, y):
        # Fit and transform X
        self.vectorizer_ = CountVectorizer(lowercase=False,
                                           tokenizer=self.tokens_to_str,
                                           ngram_range=(1, 2))
        X = self.vectorizer_.fit_transform(X)
        self.n_features_ = len(self.vectorizer_.vocabulary_)

        # Fit and transform y
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)
        self.n_classes_ = len(self.label_encoder_.classes_)

        return X, y

    def apply_transformers(self, X, y):
        X = self.vectorizer_.transform(X)
        y = self.label_encoder_.transform(y)
        return X, y

    def predict(self, X):
        X = self.vectorizer_.transform(X)
        predictions = list(self.classifier_.predict(self.predict_input_fn(X)))
        classes = np.array([p["class_ids"][0] for p in predictions])
        labels = self.label_encoder_.inverse_transform(classes)
        return labels

    def score(self, X, y, print_report=False):
        predictions = self.predict(X)
        if print_report:
            print(confusion_matrix(y, predictions))
            print(classification_report(y, predictions, digits=3))
        return f1_score(y, predictions, average="macro")

    def create_dnn_classifier(self):
        # Columns of X
        self.feature_columns_ = [tf.feature_column.numeric_column(
            key=self.features_key, shape=self.n_features_
        )]

        # Column of weights which will be used to compute loss
        weight_column = tf.feature_column.numeric_column(self.weight_key)

        # Model
        return tf.estimator.DNNClassifier(
            feature_columns=self.feature_columns_,
            weight_column=weight_column,
            hidden_units=self.hidden_units,
            n_classes=self.n_classes_,
            model_dir=self.model_dir
        )

    def train_input_fn(self, X, y, num_epochs):
        n = X.shape[0]
        num_batches = n // self.batch_size

        def input_fn():
            # Convert to tensors
            tf_X = self.convert_sparse_matrix_to_sparse_tensor(X)
            tf_y = tf.convert_to_tensor(y)

            # Batch iterator
            i = tf.train.range_input_producer(
                limit=num_batches, num_epochs=num_epochs, shuffle=False
            ).dequeue()

            # Slice sparse tensor using batch number, then convert to dense
            tf_X = tf.sparse.to_dense(tf.sparse.slice(
                tf_X, start=[i * self.batch_size, 0],
                size=[self.batch_size, self.n_features_]
            ), validate_indices=False)

            # Slice labels tensor using batch number
            output_y = tf_y[i * self.batch_size:(i + 1) * self.batch_size]

            # Create weights tensor using given weight of class "M"
            class_M = self.label_encoder_.transform(["M"])
            weights = tf.cast(tf.equal(output_y, class_M), tf.float64)
            weights = tf.multiply(weights, (self.weight_class_M - 1)) + 1

            # Return tensors in correct format
            return {self.features_key: tf_X, self.weight_key: weights}, output_y

        return input_fn

    def predict_input_fn(self, X):
        n = X.shape[0]

        # Ensure num_batches has the right value so that we predict every point
        if n % self.batch_size == 0:
            num_batches = n // self.batch_size
        else:
            num_batches = n // self.batch_size + 1

        def input_fn():
            # Convert to tensor
            tf_X = self.convert_sparse_matrix_to_sparse_tensor(X)

            # Batch iterator
            i = tf.train.range_input_producer(
                limit=num_batches, num_epochs=1, shuffle=False
            ).dequeue()

            # Slice sparse tensor using batch number, then convert to dense
            tf_X = tf.sparse.to_dense(tf.sparse.slice(
                tf_X, start=[i * self.batch_size, 0],
                size=[self.batch_size, self.n_features_]
            ), validate_indices=False)

            # Return tensor in correct format
            return {self.features_key: tf_X}

        return input_fn

    def check_warm_start(self, warm_start):
        if warm_start:
            # Check if model was already fitted
            try:
                self.classifier_
            except:
                warm_start = False
        return warm_start

    def set_model_directory(self, checkpoints_dir, model_name):
        if model_name is not None:
            self.model_dir = checkpoints_dir + model_name
            # Check model_dir doesn't already exist
            if os.path.exists(self.model_dir):
                raise ValueError("model_dir already exists")
        else:
            self.model_dir = None

    @staticmethod
    def tokens_to_str(tokens):
        # Used in CountVectorizer because we already have tokens
        return list(map(str, tokens))

    @staticmethod
    def convert_sparse_matrix_to_sparse_tensor(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensorValue(indices, coo.data, coo.shape)
