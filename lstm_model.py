import os
from collections import Counter

import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class LSTMModel(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 sentence_length,
                 embeddings,
                 num_units=50,
                 batch_size=128,
                 learning_rate=0.01,
                 dropout_keep_prob=1.0,
                 weight_class_M=1.0,
                 model_name=None,
                 checkpoints_dir="../checkpoints/",
                 ):
        self.sentence_length = sentence_length
        self.embeddings = embeddings
        self.embedding_dim = self.embeddings.shape[1]
        self.num_units = num_units
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.weight_class_M = weight_class_M
        self.features_key = "x"
        self.seq_lengths_key = "lengths"
        self.model_dir = self.set_model_directory(checkpoints_dir, model_name)

    @staticmethod
    def set_model_directory(checkpoints_dir, model_name):
        if model_name is not None:
            model_dir = checkpoints_dir + model_name
            # Check model_dir doesn't already exist
            if os.path.exists(model_dir):
                raise ValueError("model_dir already exists")
        else:
            model_dir = None
        return model_dir

    def check_warm_start(self, warm_start):
        if warm_start:
            # Check if model was already fitted
            try:
                self.classifier_
            except AttributeError:
                warm_start = False
        return warm_start

    @staticmethod
    def f1_metric_fn(labels, predictions):
        p, p_op = tf.metrics.precision(labels=labels, predictions=predictions)
        r, r_op = tf.metrics.recall(labels=labels, predictions=predictions)
        return 2 * p * r / (p + r), tf.group(p_op, r_op)

    def f1_score(self, labels, predictions):
        return {"f1-score": self.f1_metric_fn(labels=labels, predictions=predictions)}

    def network_fn(self, features):
        # Create embedding matrix
        embeddings = tf.convert_to_tensor(self.embeddings)
        unknown_words_embedding = tf.Variable(tf.random_uniform(
            [1, self.embedding_dim], -1.0, 1.0, tf.float64), trainable=True)
        embeddings = tf.concat([embeddings, unknown_words_embedding], axis=0)

        # Extract sequences embeddings
        sequences = features[self.features_key]
        embeddings = tf.nn.embedding_lookup(embeddings, tf.cast(sequences, tf.int64))

        # Extract sequences lengths
        cur_batch_size = tf.shape(embeddings)[0]
        lengths = features[self.seq_lengths_key]
        lengths = tf.cast(tf.reshape(lengths, [cur_batch_size]), tf.int32)

        # LSTM layer with dropout on outputs
        cell = tf.nn.rnn_cell.LSTMCell(self.num_units, activation="relu")
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        initial_state = cell.zero_state(cur_batch_size, tf.float64)
        outputs, _ = tf.nn.dynamic_rnn(cell, embeddings, initial_state=initial_state,
                                       sequence_length=lengths)

        # Get last relevant output
        flat_outputs = tf.reshape(outputs, [-1, self.num_units])
        relevant_index = tf.range(0, cur_batch_size) * self.sentence_length + (lengths - 1)
        relevant_output = tf.gather(flat_outputs, relevant_index)

        # Classifier
        logits = tf.layers.dense(relevant_output, self.n_classes_, activation=None)
        return logits

    def model_fn(self, features, labels, mode):
        # Network
        logits = self.network_fn(features)

        # Predict
        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predicted_classes)

        # Loss
        class_M = self.label_encoder_.transform(["M"])
        weights = tf.cast(tf.equal(labels, class_M), tf.float64)
        weights = tf.multiply(weights, (self.weight_class_M - 1)) + 1
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)

        # Eval
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, predictions=predicted_classes)

        # Train
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def input_fn(self, mode, X, y=None, num_epochs=1):
        if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            shuffle = True
        else:
            shuffle, num_epochs, y = (False, 1, None)

        sequences_lengths = X.shape[1] - np.argmax(X[:, ::-1] != 0, axis=1)
        X = {self.features_key: X, self.seq_lengths_key: sequences_lengths}

        return tf.estimator.inputs.numpy_input_fn(X, y, batch_size=self.batch_size,
                                                  num_epochs=num_epochs, shuffle=shuffle)

    def create_classifier(self):
        run_config = tf.estimator.RunConfig(model_dir=self.model_dir, log_step_count_steps=50)
        model = tf.estimator.Estimator(model_fn=self.model_fn, config=run_config)
        model = tf.contrib.estimator.add_metrics(model, self.f1_score)
        return model
    
    def pad_sentences(self, sentences):
        return pad_sequences(sentences, self.sentence_length, padding="post")

    def apply_transformers(self, X, y):
        X = self.pad_sentences(X)
        y = self.label_encoder_.transform(y)
        return X, y

    def fit_and_apply_transformers(self, X, y):
        X = self.pad_sentences(X)
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)
        self.n_classes_ = len(self.label_encoder_.classes_)
        return X, y

    def score(self, X, y):
        X, y = self.apply_transformers(X, y)
        results = self.classifier_.evaluate(self.input_fn(tf.estimator.ModeKeys.EVAL, X, y))
        return results["f1-score"]

    def predict(self, X):
        X = self.pad_sentences(X)
        classes = list(self.classifier_.predict(self.input_fn(tf.estimator.ModeKeys.PREDICT, X)))
        labels = self.label_encoder_.inverse_transform(classes)
        return labels

    def fit(self, X, y, num_epochs=1, warm_start=True):
        warm_start = self.check_warm_start(warm_start)
        if not warm_start:
            X, y = self.fit_and_apply_transformers(X, y)
            self.classifier_ = self.create_classifier()
        else:
            X, y = self.apply_transformers(X, y)

        self.classifier_.train(self.input_fn(tf.estimator.ModeKeys.TRAIN, X, y, num_epochs))
        return self
