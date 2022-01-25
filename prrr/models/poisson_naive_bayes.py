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
