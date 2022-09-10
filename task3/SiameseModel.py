from keras import layers
from keras import Model
from keras import metrics
import tensorflow as tf
from keras.models import Model
import keras.backend as K
import numpy as np


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5, scale=32):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.scale = scale
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        # Compute accuracy
        # Return 1 if the distance between the anchor and the positive is smaller than the distance between the anchor and the negative, 0 otherwise.
        ap_distance, an_distance, anchor, positive, negative = self.siamese_network(data)
        accuracy = tf.cast(ap_distance < an_distance, tf.float32)
        
        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(accuracy)
        return {"accuracy": self.loss_tracker.result()}
        
    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance, q, p, n = self.siamese_network(data)

        # mean_distance = (ap_distance + an_distance) / 2
        # mult1 = tf.ones_like(mean_distance) + tf.cast(tf.greater(ap_distance, tf.ones_like(mean_distance) * 1.5), tf.float32)*1.0
        # mult2 = tf.ones_like(mean_distance) + tf.cast(tf.greater(tf.ones_like(mean_distance) * 0.5, an_distance), tf.float32)*1.0


        # loss = ap_distance*mult1 - an_distance*mult2
        # loss = tf.maximum(loss + self.margin, 0.0)
        # return loss


        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

        # return ap_distance if np.random.uniform() > 0.5 else -an_distance

        # self.similarity = 'cos'

        # if self.similarity == 'dot':
        #     sim_p = self.dot_similarity(q, p)
        #     sim_n = self.dot_similarity(q, n)
        # elif self.similarity == 'cos':
        #     sim_p = self.cosine_similarity(q, p)
        #     sim_n = self.cosine_similarity(q, n)       
        # else:
        #     raise ValueError('This similarity is not implemented.')

        # alpha_p = K.relu(-sim_p + 1 + self.margin)
        # alpha_n = K.relu(sim_n + self.margin)
        # margin_p = 1 - self.margin
        # margin_n = self.margin

        # logit_p = tf.reshape(-self.scale * alpha_p * (sim_p - margin_p), (-1, 1))
        # logit_n = tf.reshape(self.scale * alpha_n * (sim_n - margin_n), (-1, 1))

        # label_p = tf.ones_like(logit_p)
        # label_n = tf.zeros_like(logit_n)

        # return K.mean(metrics.binary_crossentropy(tf.concat([label_p, label_n], axis=0), tf.concat([logit_p, logit_n], axis=0),from_logits=True))



    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]








    def dot_similarity(self, x, y):
        x = K.reshape(x, (K.shape(x)[0], -1))
        y = K.reshape(y, (K.shape(y)[0], -1))
        return K.dot(x, K.transpose(y))
    
    def cosine_similarity(self, x, y):
        x = K.reshape(x, (K.shape(x)[0], -1))
        y = K.reshape(y, (K.shape(y)[0], -1))
        abs_x = K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
        abs_y = K.sqrt(K.sum(K.square(y), axis=1, keepdims=True))
        up = K.dot(x, K.transpose(y))
        down = K.dot(abs_x, K.transpose(abs_y))
        return up / down