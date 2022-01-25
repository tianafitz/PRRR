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


<<<<<<< HEAD:prrr/models/poisson_naive_bayes.py
=======
# ------- Specify model ---------


def rrr(X, q, k, size_factors=None):

    n, p = X.shape

    A = yield tfd.Gamma(
        concentration=tf.fill([p, k], 2.0), rate=tf.ones([p, k]), name="A"
    )

    B = yield tfd.Gamma(
        concentration=tf.fill([k, q], 2.0), rate=tf.ones([k, q]), name="B"
    )

    count_mean = tf.matmul(tf.matmul(X.astype("float32"), A), B)
    # import ipdb; ipdb.set_trace()
    if size_factors is not None:
        count_mean = tf.multiply(count_mean, size_factors)
        
    Y = yield tfd.Poisson(
        rate=count_mean, name="Y"
    )


def fit_rrr(X, Y, k, size_factors=None):

    assert X.shape[0] == Y.shape[0]
    n, p = X.shape
    _, q = Y.shape

    # ------- Specify model ---------

    rrr_model = functools.partial(rrr, X=X, q=q, k=k, size_factors=size_factors)

    model = tfd.JointDistributionCoroutineAutoBatched(rrr_model)

    def target_log_prob_fn(A, B):
        return model.log_prob((A, B, Y))

    # ------- Specify variational families -----------

    # Variational parameter means

    qA_mean = tf.Variable(tf.random.normal([p, k]))
    qA_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([p, k]), bijector=tfb.Softplus()
    )

    qB_mean = tf.Variable(tf.random.normal([k, q]))
    qB_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([k, q]), bijector=tfb.Softplus()
    )

    def factored_normal_variational_model():
        qA = yield tfd.LogNormal(loc=qA_mean, scale=qA_stddv, name="qA")
        qB = yield tfd.LogNormal(loc=qB_mean, scale=qB_stddv, name="qB")

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

    return return_dict


def poisson_regression(X, q):

    n, p = X.shape

    # B = yield tfd.Gamma(
    #     concentration=tf.fill([p, q], 2.0), rate=tf.ones([p, q]), name="B"
    # )
    B = yield tfd.Normal(
        loc=tf.zeros([p, q]), scale=np.sqrt(1/(p * q)) * tf.ones([p, q]), name="B"
    )

    predicted_rate = tf.exp(tf.matmul(X.astype("float32"), B))

    Y = yield tfd.Poisson(
        rate=predicted_rate, name="Y"
    )


def fit_poisson_regression(X, Y, use_vi=True, k_initialize=2):

    assert X.shape[0] == Y.shape[0]
    n, p = X.shape
    _, q = Y.shape

    # ------- Specify model ---------

    pr_model = functools.partial(poisson_regression, X=X, q=q)

    model = tfd.JointDistributionCoroutineAutoBatched(pr_model)

    def target_log_prob_fn(B):
        return model.log_prob((B, Y))

    if use_vi:

        # ------- Specify variational families -----------

        qB_mean = tf.Variable(tf.random.normal([p, q], stddev=np.sqrt(1/(p * q))))
        qB_stddv = tfp.util.TransformedVariable(
            1e-4 * tf.ones([p, q]), bijector=tfb.Softplus()
        )

        def factored_normal_variational_model():
            qB = yield tfd.LogNormal(loc=qB_mean, scale=qB_stddv, name="qB")

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
            "B_mean": qB_mean,
            "B_stddev": qB_stddv
        }

    else:
        ## MAP estimation

        B = tf.Variable(tf.random.normal([p, q]))
        # A_tmp = tf.Variable(tf.random.normal([p, k_initialize], stddev=np.sqrt(1 / p)))
        # B_tmp = tf.Variable(tf.random.normal([k_initialize, q], stddev=np.sqrt(1 / q)))
        # B = tf.matmul(A_tmp, B_tmp)

        losses = tfp.math.minimize(
            lambda: -target_log_prob_fn(B),
            optimizer=tf.optimizers.Adam(learning_rate=0.05),
            num_steps=500)

        return_dict = {
            "loss_trace": losses,
            "B": B,
        }

    return return_dict


>>>>>>> 102806d0d914a9f0faa900640814f2292f6b32c0:models/prrr_nb_tfp.py
def naive_bayes(X, q):

    n, p = X.shape

    lam = yield tfd.Gamma(
        concentration=tf.fill([p, q], 2.0), rate=tf.ones([p, q]), name="A"
    )

    Y = yield tfd.Poisson(rate=tf.matmul(X.astype("float32"), lam), name="Y")


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

    qlam_concentration = tf.fill([p, q], 2.0)
    qlam_rate = tfp.util.TransformedVariable(tf.ones([p, q]), bijector=tfb.Softplus())

    def factored_normal_variational_model():
        qlam = yield tfd.Gamma(
            concentration=qlam_concentration, rate=qlam_rate, name="qlam"
        )

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
    }

    return return_dict


if __name__ == "__main__":

    n = 200
    p = 4
    q = 20
    k_true = 5
    cell_types_ints = np.random.choice(np.arange(p), size=n, replace=True)
    X = np.zeros((n, p))
    for ii in range(n):
        X[ii, cell_types_ints[ii]] = 1
    A_true = np.random.gamma(2, 1, size=(p, k_true))
    B_true = np.random.gamma(2, 1, size=(k_true, q))
    Y_mean = X @ A_true @ B_true
    Y = np.random.poisson(Y_mean)

    all_rrr = []
    all_nb = []
    k_range = [1, 3, 4, 5, 6, 10, 20]
    for k_rrr in k_range:
        rrr_elbo_list = []
        nb_elbo_list = []

        for i in range(3):
            rrr_results = fit_rrr(X=X, Y=Y, k=k_rrr)
            rrr_elbo = -rrr_results["loss_trace"].numpy()[-1]
            print("RRR ELBO: {}".format(rrr_elbo))
            rrr_elbo_list.append(rrr_elbo)

            nb_results = fit_naive_bayes(X=X, Y=Y)
            nb_elbo = -nb_results["loss_trace"].numpy()[-1]
            print("Naive Bayes ELBO: {}".format(nb_elbo))
            nb_elbo_list.append(nb_elbo)
        all_rrr.append(rrr_elbo_list)
        all_nb.append(nb_elbo_list)

    plt.plot(k_range, np.mean(all_rrr, axis=1), "o-", label="RRR")
    plt.plot(k_range, np.mean(all_nb, axis=1), "o-", label="Naive Bayes")
    plt.axvline(k_true, linestyle="--", label="True rank")
    plt.legend()
    plt.xlabel("Rank for RRR")
    plt.ylabel("ELBO (higher is better)")
    plt.tight_layout()
    plt.show()

    # import ipdb; ipdb.set_trace()
