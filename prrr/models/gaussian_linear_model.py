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
LEARNING_RATE_VI = 1e-1


def gaussian_regression(X, q):

    n, p = X.shape

    B = yield tfd.Normal(
        loc=tf.zeros([p, q]), scale=np.sqrt(1 / (p * q)) * tf.ones([p, q]), name="B"
    )

    predicted_mean = tf.matmul(X.astype("float32"), B)

    standard_deviation = yield tfd.InverseGamma(
        concentration=1.0, scale=1.0, name="sigma2"
    )

    Y = yield tfd.Normal(loc=predicted_mean, scale=standard_deviation, name="Y")


def fit_lm(X, Y, use_vi=True):

    assert X.shape[0] == Y.shape[0]
    n, p = X.shape
    _, q = Y.shape

    # ------- Specify model ---------

    lr_model = functools.partial(gaussian_regression, X=X, q=q)

    model = tfd.JointDistributionCoroutineAutoBatched(lr_model)

    def target_log_prob_fn(B, sigma2):
        return model.log_prob((B, sigma2, Y))

    if use_vi:

        # ------- Specify variational families -----------

        qB_mean = tf.Variable(tf.random.normal([p, q], stddev=1))
        qB_stddv = tfp.util.TransformedVariable(
            1e-4 * tf.ones([p, q]), bijector=tfb.Softplus()
        )

        qsigma2_concentration = tfp.util.TransformedVariable(
            tf.ones([1]), bijector=tfb.Softplus()
        )
        qsigma2_scale = tfp.util.TransformedVariable(
            tf.ones([1]), bijector=tfb.Softplus()
        )

        def factored_normal_variational_model():
            qB = yield tfd.Normal(loc=qB_mean, scale=qB_stddv, name="qB")
            qsigma2 = yield tfd.InverseGamma(
                concentration=qsigma2_concentration, scale=qsigma2_scale, name="qsigma2"
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

        return_dict = {"loss_trace": losses, "B_mean": qB_mean, "B_stddev": qB_stddv}
        # plt.plot(return_dict["loss_trace"])
        # plt.show()
        # print(return_dict)
        # import ipdb; ipdb.set_trace()

    # else:
    #     ## MAP estimation

    #     B = tf.Variable(tf.random.normal([p, q]))
    #     # A_tmp = tf.Variable(tf.random.normal([p, k_initialize], stddev=np.sqrt(1 / p)))
    #     # B_tmp = tf.Variable(tf.random.normal([k_initialize, q], stddev=np.sqrt(1 / q)))
    #     # B = tf.matmul(A_tmp, B_tmp)

    #     losses = tfp.math.minimize(
    #         lambda: -target_log_prob_fn(B),
    #         optimizer=tf.optimizers.Adam(learning_rate=0.05),
    #         num_steps=500,
    #     )

    #     return_dict = {
    #         "loss_trace": losses,
    #         "B": B,
    #     }

    return return_dict


if __name__ == "__main__":

    n = 30
    p = 10
    q = 1
    # cell_types_ints = np.random.choice(np.arange(p), size=n, replace=True)
    # X = np.zeros((n, p))
    # for ii in range(n):
    #     X[ii, cell_types_ints[ii]] = 1

    n_repeats = 10
    B_estimated = []
    B_trues = []
    for _ in range(n_repeats):
        X = np.random.uniform(low=-3, high=3, size=(n, p))
        B_true = np.random.normal(0, 1, size=(p, q))
        Y_mean = X @ B_true
        Y = np.random.normal(loc=Y_mean)

        rrr_results = fit_lm(X=X, Y=Y)

        B_estimated.append(rrr_results["B_mean"].numpy().squeeze())
        B_trues.append(B_true.squeeze())

    # print("True B: ", B_true)
    # print("Estimated B: ", rrr_results["B_mean"])
    plt.scatter(B_estimated, B_trues)
    plt.show()
    import ipdb

    ipdb.set_trace()
