import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb


from scipy.stats import multivariate_normal


tf.enable_v2_behavior()

warnings.filterwarnings("ignore")

NUM_VI_ITERS = 1000
LEARNING_RATE_VI = 0.01


# ------- Specify model ---------


def grrr(X, q, k, size_factors=None):

    n, p = X.shape

    A = yield tfd.Normal(
        loc=tf.zeros([p, k]), scale=np.sqrt(1 / p) * tf.ones([p, k]), name="A"
    )

    B = yield tfd.Normal(
        loc=tf.zeros([k, q]), scale=np.sqrt(1 / q) * tf.ones([k, q]), name="B"
    )

    count_mean_log = tf.matmul(tf.matmul(X.astype("float32"), A), B)
    count_mean = tf.exp(count_mean_log)

    if size_factors is not None:
        count_mean = tf.multiply(count_mean, size_factors)
        
    Y = yield tfd.Poisson(
        rate=count_mean, name="Y"
    )


def fit_grrr(X, Y, k, size_factors=None, use_vi=True):

    assert X.shape[0] == Y.shape[0]
    n, p = X.shape
    _, q = Y.shape

    # ------- Specify model ---------

    rrr_model = functools.partial(grrr, X=X, q=q, k=k, size_factors=size_factors)

    model = tfd.JointDistributionCoroutineAutoBatched(rrr_model)

    def target_log_prob_fn(A, B):
        return model.log_prob((A, B, Y))

    if use_vi:

        # ------- Specify variational families -----------

        # Variational parameter means

        qA_mean = tf.Variable(tf.random.normal([p, k], stddev=np.sqrt(1 / p)))
        qA_stddv = tfp.util.TransformedVariable(
            1e-4 * tf.ones([p, k]), bijector=tfb.Softplus()
        )

        qB_mean = tf.Variable(tf.random.normal([k, q], stddev=np.sqrt(1 / q)))
        qB_stddv = tfp.util.TransformedVariable(
            1e-4 * tf.ones([k, q]), bijector=tfb.Softplus()
        )

        def factored_normal_variational_model():
            qA = yield tfd.Normal(loc=qA_mean, scale=qA_stddv, name="qA")
            qB = yield tfd.Normal(loc=qB_mean, scale=qB_stddv, name="qB")

        # Surrogate posterior that we will try to make close to p
        surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
            factored_normal_variational_model
        )

        # --------- Fit variational inference model using MC samples and gradient descent ----------

        losses = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn,
            surrogate_posterior=surrogate_posterior,
            optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE_VI),
            num_steps=NUM_VI_ITERS,
        )

        return_dict = {
            "loss_trace": losses,
            "A_mean": qA_mean,
            "A_stddev": qA_stddv,
            "B_mean": qB_mean,
            "B_stddev": qB_stddv,
        }

    else:
        ## MAP estimation

        A = tf.Variable(tf.random.normal([p, k], stddev=np.sqrt(1 / p)))
        B = tf.Variable(tf.random.normal([k, q], stddev=np.sqrt(1 / q)))

        losses = tfp.math.minimize(
            lambda: -target_log_prob_fn(A, B),
            optimizer=tf.optimizers.Adam(learning_rate=0.01),
            num_steps=500)

        return_dict = {
            "loss_trace": losses,
            "A": A,
            "B": B,
        }

    return return_dict


if __name__ == "__main__":

    pass
