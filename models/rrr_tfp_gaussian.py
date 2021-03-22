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

NUM_VI_ITERS = 300
LEARNING_RATE_VI = 0.05


# ------- Specify model ---------


def rrr(X, q, k):

    n, p = X.shape

    A = yield tfd.Normal(loc=tf.zeros([p, k]), scale=tf.ones([p, k]), name="A")

    B = yield tfd.Normal(loc=tf.zeros([k, q]), scale=tf.ones([k, q]), name="B")

    Y = yield tfd.Normal(
        loc=tf.matmul(tf.matmul(X.astype("float32"), A), B),
        scale=tf.ones([n, q]),
        name="Y",
    )


def fit_rrr(X, Y, k):

    assert X.shape[0] == Y.shape[0]
    n, p = X.shape
    _, q = Y.shape

    # ------- Specify model ---------

    rrr_model = functools.partial(rrr, X=X, q=q, k=k)

    model = tfd.JointDistributionCoroutineAutoBatched(rrr_model)

    def target_log_prob_fn(A, B):
        return model.log_prob((A, B, Y))

    # ------- Specify variational families -----------

    # Variational parmater means

    qA_mean = tf.Variable(tf.random.normal([p, k]))
    qA_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([p, k]), bijector=tfb.Softplus()
    )

    qB_mean = tf.Variable(tf.random.normal([k, q]))
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

    return_dict = {"loss_trace": losses, "A_mean": qA_mean, "B_mean": qB_mean}

    return return_dict


def naive_bayes(X, q):

    n, p = X.shape

    lam = yield tfd.Normal(loc=tf.zeros([p, q]), scale=tf.ones([p, q]), name="A")

    Y = yield tfd.Normal(
        loc=tf.matmul(X.astype("float32"), lam), scale=tf.ones([n, q]), name="Y"
    )


def fit_naive_bayes(X, Y):

    assert Y.shape[0] == X.shape[0]
    n, p = X.shape
    q = Y.shape[1]

    # ------- Specify model ---------

    nb_model = functools.partial(naive_bayes, X=X, q=q)

    model = tfd.JointDistributionCoroutineAutoBatched(nb_model)

    def target_log_prob_fn(lam):
        return model.log_prob((lam, Y))

    # ------- Specify variational families -----------

    # Variational parmater means

    qlam_mean = tf.Variable(tf.random.normal([p, q]))
    qlam_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([p, q]), bijector=tfb.Softplus()
    )

    def factored_normal_variational_model():
        qlam = yield tfd.Normal(loc=qlam_mean, scale=qlam_stddv, name="qlam")

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
        "lam_mean": qlam_mean,
    }

    # plt.scatter(A_true[:, 0], qA_mean.numpy()[:, 0])
    # plt.show()
    # import ipdb; ipdb.set_trace()

    return return_dict


if __name__ == "__main__":

    n = 100
    p = 10
    q = 20
    k_true = 5
    cell_types_ints = np.random.choice(np.arange(p), size=n, replace=True)
    X = np.zeros((n, p))
    for ii in range(n):
        X[ii, cell_types_ints[ii]] = 1
    A_true = np.random.normal(0, 1, size=(p, k_true))
    B_true = np.random.normal(0, 1, size=(k_true, q))
    Y_mean = X @ A_true @ B_true
    Y = np.random.normal(loc=Y_mean)

    rrr_elbo_list = []
    nb_elbo_list = []
    k_range = [1, 3, 4, 5, 6, 10, 20]
    for k_rrr in k_range:

        rrr_results = fit_rrr(X=X, Y=Y, k=k_rrr)
        rrr_elbo = -rrr_results["loss_trace"].numpy()[-1]
        print("RRR ELBO: {}".format(rrr_elbo))
        rrr_elbo_list.append(rrr_elbo)

        nb_results = fit_naive_bayes(X=X, Y=Y)
        nb_elbo = -nb_results["loss_trace"].numpy()[-1]
        print("Naive Bayes ELBO: {}".format(nb_elbo))
        nb_elbo_list.append(nb_elbo)

    plt.plot(k_range, rrr_elbo_list, "o-", label="RRR")
    plt.plot(k_range, nb_elbo_list, "o-", label="Naive Bayes")
    plt.axvline(k_true, linestyle="--", label="True rank")
    plt.legend()
    plt.xlabel("Rank for RRR")
    plt.ylabel("ELBO (higher is better)")
    plt.tight_layout()
    plt.show()

    import ipdb

    ipdb.set_trace()
