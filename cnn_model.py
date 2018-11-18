import os

import tensorflow as tf
from tensorflow.python.keras._impl.keras.preprocessing.sequence import pad_sequences

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class CNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 sentence_length,
                 embeddings,
                 filters_by_ksize=5,
                 kernel_sizes=(2,),
                 batch_size=128,
                 learning_rate=0.1,
                 dropout_keep_prob=1.0,
                 model_name=None,
                 checkpoints_dir="../checkpoints/",
                 ):
        self.sentence_length = sentence_length
        self.embeddings = embeddings
        self.embedding_dim = self.embeddings.shape[1]
        self.filters_by_ksize = filters_by_ksize
        self.kernel_sizes = kernel_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.features_key = "x"
        self.weight_key = "weight"
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
            except NameError:
                warm_start = False
        return warm_start

    @staticmethod
    def f1_metric_fn(labels, predictions):
        p, p_op = tf.metrics.precision(labels=labels, predictions=predictions)
        r, r_op = tf.metrics.recall(labels=labels, predictions=predictions)
        return 2 * p * r / (p + r), tf.group(p_op, r_op)

    def f1_score(self, labels, predictions):
        return {"f1-score": self.f1_metric_fn(labels=labels, predictions=predictions)}

    def network_fn(self, features, params):
        # Create embedding matrix
        embeddings = tf.convert_to_tensor(self.embeddings)
        unknown_words_embedding = tf.Variable(tf.random_uniform(
            [1, self.embedding_dim], -1.0, 1.0, tf.float64), trainable=True)
        embeddings = tf.concat([embeddings, unknown_words_embedding], axis=0)

        # Extract sequences embeddings
        sequences = tf.feature_column.input_layer(features, params['feature_columns'])
        embeddings = tf.nn.embedding_lookup(embeddings, tf.cast(sequences, tf.int64))

        # Convolutions and max poolings
        feature_maps = []
        iterator = zip([self.filters_by_ksize] * len(self.kernel_sizes), self.kernel_sizes)
        for filters, kernel_size in iterator:
            tmp = tf.layers.conv1d(embeddings, filters, kernel_size, padding="same")
            tmp = tf.layers.max_pooling1d(tmp, [self.sentence_length], strides=1, padding="valid")
            feature_maps.append(tmp)

        # Concat all feature maps, add dropout, and add softmax
        shape = [-1, self.filters_by_ksize * len(self.kernel_sizes)]
        feature_maps = tf.reshape(tf.concat(feature_maps, axis=2), shape)
        feature_maps = tf.nn.dropout(feature_maps, self.dropout_keep_prob)
        logits = tf.layers.dense(feature_maps, self.n_classes_, activation=None)
        return logits

    def model_fn(self, features, labels, mode, params):
        # Network
        logits = self.network_fn(features, params)

        # Predict
        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predicted_classes)

        # Loss
        class_M = self.label_encoder_.transform(["M"])
        weights = tf.cast(tf.equal(labels, class_M), tf.float64)
        weights = tf.multiply(weights, (6.63 - 1)) + 1
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
        X = {self.features_key: X}
        return tf.estimator.inputs.numpy_input_fn(X, y, self.batch_size, num_epochs, shuffle)

    def create_dnn_classifier(self):
        # Columns of X
        self.feature_columns_ = [tf.feature_column.numeric_column(
            key=self.features_key, shape=self.sentence_length)]

        # Params
        params = {"feature_columns": self.feature_columns_, "n_classes": self.n_classes_}
        run_config = tf.estimator.RunConfig(model_dir=self.model_dir, log_step_count_steps=10)
        
        # Model
        model = tf.estimator.Estimator(model_fn=self.model_fn, params=params, config=run_config)
        model = tf.contrib.estimator.add_metrics(model, self.f1_score)
        return model

    def apply_transformers(self, X, y):
        X = pad_sequences(X, self.sentence_length)
        y = self.label_encoder_.transform(y)
        return X, y

    def fit_and_apply_transformers(self, X, y):
        X = pad_sequences(X, self.sentence_length)
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)
        self.n_classes_ = len(self.label_encoder_.classes_)
        return X, y

    def fit(self, X, y, num_epochs=1, warm_start=True):
        warm_start = self.check_warm_start(warm_start)
        if not warm_start:
            X, y = self.fit_and_apply_transformers(X, y)
            self.classifier_ = self.create_dnn_classifier()
        else:
            X, y = self.apply_transformers(X, y)

        self.classifier_.train(self.input_fn(tf.estimator.ModeKeys.TRAIN, X, y, num_epochs))
        return self

    def predict(self, X):
        X = pad_sequences(X, self.sentence_length)
        classes = list(self.classifier_.predict(self.input_fn(tf.estimator.ModeKeys.PREDICT, X)))
        labels = self.label_encoder_.inverse_transform(classes)
        return labels

    def score(self, X, y):
        X, y = self.apply_transformers(X, y)
        results = self.classifier_.evaluate(self.input_fn(tf.estimator.ModeKeys.EVAL, X, y))
        return results["f1-score"]
