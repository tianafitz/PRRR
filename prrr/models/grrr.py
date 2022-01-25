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

NUM_VI_ITERS = 2000
LEARNING_RATE = 1e-2


class GRRR:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def grrr_model(self, X, log_size_factors):

        A = yield tfd.Normal(
            loc=tf.zeros([self.p, self.latent_dim]),
            scale=tf.ones([self.p, self.latent_dim]),
            name="A",
        )

        B = yield tfd.Normal(
            loc=tf.zeros([self.latent_dim, self.q]),
            scale=tf.ones([self.latent_dim, self.q]),
            name="B",
        )

        linear_predictor = tf.matmul(tf.matmul(X, A), B)

        # if log_size_factors is not None:
        #     linear_predictor = tf.add(linear_predictor, log_size_factors)

        predicted_rate = tf.exp(linear_predictor)

        Y = yield tfd.Poisson(rate=predicted_rate, name="Y")

    def fit(self, X, Y, size_factors=None, use_total_counts_as_size_factors=True, use_vi=True):

        assert X.shape[0] == Y.shape[0]
        self.n, self.p = X.shape
        _, self.q = Y.shape
        self.size_factors = size_factors
        X = X.astype("float32")

        if use_total_counts_as_size_factors and size_factors is None:
            size_factors = np.sum(Y, axis=1).astype(float).reshape(-1, 1)
        log_size_factors = np.log(size_factors)

        # ------- Specify model ---------

        rrr_model = functools.partial(self.grrr_model, X=X, log_size_factors=log_size_factors)

        model = tfd.JointDistributionCoroutineAutoBatched(rrr_model)

        def target_log_prob_fn(A, B):
            return model.log_prob((A, B, Y))

        if use_vi:

            # ------- Specify variational families -----------

            # Variational parameter means

            qA_mean = tf.Variable(
                tf.random.normal(
                    [self.p, self.latent_dim], stddev=np.sqrt(1 / self.p)
                )
            )
            qA_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([self.p, self.latent_dim]), bijector=tfb.Softplus()
            )

            qB_mean = tf.Variable(
                tf.random.normal(
                    [self.latent_dim, self.q], stddev=np.sqrt(1 / self.q)
                )
            )
            qB_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([self.latent_dim, self.q]), bijector=tfb.Softplus()
            )

            def factored_normal_variational_model():
                qA = yield tfd.Normal(loc=qA_mean, scale=qA_stddv, name="qA")
                qB = yield tfd.Normal(loc=qB_mean, scale=qB_stddv, name="qB")

            # Surrogate posterior that we will try to make close to p
            surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
                factored_normal_variational_model
            )

            # --------- Fit variational inference model using MC samples and gradient descent ----------

            convergence_criterion = (
                tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=1e-1)
            )
            optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

            losses = tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn,
                surrogate_posterior=surrogate_posterior,
                optimizer=optimizer,
                num_steps=NUM_VI_ITERS,
                convergence_criterion=convergence_criterion,
            )
            n_iters = optimizer._iterations.numpy()

            self.loss_trace = losses[:n_iters]

            self.param_dict = {
                "A_mean": qA_mean,
                "A_stddev": qA_stddv,
                "B_mean": qB_mean,
                "B_stddev": qB_stddv,
            }

        else:
            ## MAP estimation

            A = tf.Variable(
                tf.random.normal(
                    [self.p, self.latent_dim] #, stddev=np.sqrt(1 / self.p)
                )
            )
            B = tf.Variable(
                tf.random.normal(
                    [self.latent_dim, self.q] #, stddev=np.sqrt(1 / self.q)
                )
            )

            # convergence_criterion = (
            #     tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=1e-1)
            # )
            optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

            losses = tfp.math.minimize(
                lambda: -target_log_prob_fn(A, B),
                optimizer=optimizer,
                num_steps=NUM_VI_ITERS,
                # convergence_criterion=convergence_criterion,
            )
            

            self.loss_trace = losses #[:n_iters]

            self.param_dict = {
                "A": A,
                "B": B,
            }

        return


if __name__ == "__main__":

    n = 1000
    p = 5
    q = 1000
    k_true = 1

    n_repeats = 10

    for _ in range(n_repeats):
        X = np.random.uniform(low=-3, high=3, size=(n, p))
        A_true = np.random.normal(0, 0.1, size=(p, k_true))
        B_true = np.random.normal(0, 0.1, size=(k_true, q))

        linear_predictor = X @ A_true @ B_true
        size_factors = np.random.uniform(low=1e-1, high=2, size=n).reshape(-1, 1)
        linear_predictor += np.log(size_factors)
        Y_mean = np.exp(linear_predictor)
        Y = np.random.poisson(Y_mean)

        grrr = GRRR(latent_dim=k_true)
        grrr.fit(X=X, Y=Y, use_vi=False, size_factors=size_factors)

        A_est, B_est = grrr.param_dict["A"].numpy(), grrr.param_dict["B"].numpy()
        preds = np.exp(X @ A_est @ B_est + np.log(size_factors))

        plt.scatter(Y_mean[:, 0], preds[:, 0])
        plt.show()

        # import ipdb; ipdb.set_trace()







