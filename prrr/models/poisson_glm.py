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


def poisson_regression(X, q, log_size_factors):

    n, p = X.shape

    B = yield tfd.Normal(loc=tf.zeros([p, q]), scale=tf.ones([p, q]), name="B")

    linear_predictor = tf.matmul(X, B)

    if log_size_factors is not None:
        linear_predictor = tf.add(linear_predictor, log_size_factors)

    predicted_rate = tf.exp(linear_predictor)

    Y = yield tfd.Poisson(rate=predicted_rate, name="Y")


def fit_poisson_regression(
    X,
    Y,
    use_vi=True,
    k_initialize=2,
    size_factors=None,
    use_total_counts_as_size_factors=False,
    pseudocount=1.0,
):

    assert X.shape[0] == Y.shape[0]
    n, p = X.shape
    _, q = Y.shape
    X = X.astype("float32")

    if use_total_counts_as_size_factors and size_factors is None:
        size_factors = np.sum(Y, axis=1).astype(float).reshape(-1, 1)
    log_size_factors = np.log(size_factors)

    # ------- Specify model ---------

    pr_model = functools.partial(
        poisson_regression, X=X, q=q, log_size_factors=log_size_factors
    )

    model = tfd.JointDistributionCoroutineAutoBatched(pr_model)

    def target_log_prob_fn(B):
        return model.log_prob((B, Y))

    if use_vi:

        # ------- Specify variational families -----------

        qB_mean = tf.Variable(tf.random.normal([p, q], stddev=1))
        qB_stddv = tfp.util.TransformedVariable(
            1e-4 * tf.ones([p, q]), bijector=tfb.Softplus()
        )

        def factored_normal_variational_model():
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

        return_dict = {"loss_trace": losses, "B_mean": qB_mean, "B_stddev": qB_stddv}

    else:
        ## MAP estimation

        B = tf.Variable(tf.random.normal([p, q]))

        losses = tfp.math.minimize(
            lambda: -target_log_prob_fn(B),
            optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE_VI),
            num_steps=NUM_VI_ITERS,
        )
        # import ipdb; ipdb.set_trace()

        return_dict = {
            "loss_trace": losses,
            "B": B,
        }

    return return_dict


if __name__ == "__main__":

    n = 200
    p = 1
    q = 10

    n_repeats = 10
    B_estimated = []
    B_trues = []
    for _ in range(n_repeats):
        X = np.random.uniform(low=-3, high=3, size=(n, p))
        B_true = np.random.normal(0, 1, size=(p, q))
        # Y_mean = np.exp(X @ B_true)
        linear_predictor = X @ B_true
        size_factors = np.random.uniform(low=1e-1, high=2, size=n).reshape(-1, 1)
        linear_predictor += np.log(size_factors)
        Y_mean = np.exp(linear_predictor)

        Y = np.random.poisson(lam=Y_mean)
        # import ipdb; ipdb.set_trace()

        ## Remove samples with no counts
        nonzero_idx = np.where(Y.sum(1) != 0)[0]
        Y = Y[nonzero_idx]
        Y_mean = Y_mean[nonzero_idx]
        X = X[nonzero_idx]
        size_factors = size_factors[nonzero_idx]

        rrr_results = fit_poisson_regression(
            X=X, Y=Y, use_total_counts_as_size_factors=False, use_vi=False, size_factors=size_factors
        )
        # import ipdb

        # ipdb.set_trace()

        B_est = rrr_results["B"].numpy()

        ## Make predictions
        Y_normalized = Y / Y.sum(1).reshape(-1, 1)
        # import ipdb; ipdb.set_trace()

        # import ipdb; ipdb.set_trace()
        preds = np.exp(X @ B_est + np.log(size_factors))
        # preds = np.exp(X @ B_est)
        preds_rand = np.exp(
            X @ np.random.normal(size=(p, q)) + np.log(size_factors)
        )
        # preds_rand = np.exp(X @ np.random.normal(size=(p, q)))

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(Y_mean[:, 0], preds[:, 0])
        plt.subplot(122)
        plt.scatter(Y_mean[:, 0], preds_rand[:, 0])
        plt.show()

        import ipdb

        ipdb.set_trace()

        B_estimated.append(B_est)
        B_trues.append(B_true.squeeze())
        print(rrr_results["B"].numpy().squeeze())
    plt.scatter(B_estimated, B_trues)
    plt.show()
