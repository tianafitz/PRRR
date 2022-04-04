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

LEARNING_RATE = 1e-2


class GRRR:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def grrr_model(self, X, log_size_factors):

        U = yield tfd.Normal(
            loc=tf.zeros([self.p, self.latent_dim]),
            scale=tf.ones([self.p, self.latent_dim]),
            name="U",
        )

        V = yield tfd.Normal(
            loc=tf.zeros([self.latent_dim, self.q]),
            scale=tf.ones([self.latent_dim, self.q]),
            name="V",
        )

        # Intercept
        v0 = yield tfd.Normal(
            loc=tf.zeros([1, self.q]),
            scale=tf.ones([1, self.q]),
            name="v0",
        )

        linear_predictor = tf.matmul(tf.matmul(X, U), V) + v0

        if log_size_factors is not None:
            linear_predictor = tf.add(linear_predictor, log_size_factors)

        predicted_rate = tf.exp(linear_predictor)

        Y = yield tfd.Poisson(rate=predicted_rate, name="Y")

    def fit(self, X, Y, size_factors=None, use_total_counts_as_size_factors=True, use_vi=True, n_iters=2_000, learning_rate=1e-2):

        assert X.shape[0] == Y.shape[0]
        self.n, self.p = X.shape
        _, self.q = Y.shape
        X = X.astype("float32")

        if use_total_counts_as_size_factors and size_factors is None:
            size_factors = np.sum(Y, axis=1).astype(float).reshape(-1, 1)
        log_size_factors = np.log(size_factors)
        self.size_factors = size_factors

        # ------- Specify model ---------

        rrr_model = functools.partial(self.grrr_model, X=X, log_size_factors=log_size_factors)

        model = tfd.JointDistributionCoroutineAutoBatched(rrr_model)

        def target_log_prob_fn(U, V, b0):
            return model.log_prob((U, V, b0, Y))

        if use_vi:

            # ------- Specify variational families -----------

            # Variational parameter means

            qU_mean = tf.Variable(
                tf.random.normal(
                    [self.p, self.latent_dim], stddev=np.sqrt(1 / self.p)
                )
            )
            qU_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([self.p, self.latent_dim]), bijector=tfb.Softplus()
            )

            qV_mean = tf.Variable(
                tf.random.normal(
                    [self.latent_dim, self.q], stddev=np.sqrt(1 / self.q)
                )
            )
            qV_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([self.latent_dim, self.q]), bijector=tfb.Softplus()
            )

            qv0_mean = tf.Variable(
                tf.random.normal(
                    [1, self.q], stddev=np.sqrt(1 / self.q)
                )
            )
            qv0_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([1, self.q]), bijector=tfb.Softplus()
            )

            def factored_normal_variational_model():
                qU = yield tfd.Normal(loc=qU_mean, scale=qU_stddv, name="qU")
                qV = yield tfd.Normal(loc=qV_mean, scale=qV_stddv, name="qB")
                qv0 = yield tfd.Normal(loc=qv0_mean, scale=qv0_stddv, name="qb0")

            # Surrogate posterior that we will try to make close to p
            surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
                factored_normal_variational_model
            )

            # --------- Fit variational inference model using MC samples and gradient descent ----------

            convergence_criterion = (
                tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=learning_rate)
            )
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

            losses = tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn,
                surrogate_posterior=surrogate_posterior,
                optimizer=optimizer,
                num_steps=n_iters,
                convergence_criterion=convergence_criterion,
            )
            n_iters = optimizer._iterations.numpy()

            self.loss_trace = losses[:n_iters]

            self.param_dict = {
                "U_mean": qU_mean,
                "U_stddev": qU_stddv,
                "V_mean": qV_mean,
                "V_stddev": qV_stddv,
                "v0_mean": qv0_mean,
                "v0_stddev": vb0_stddv,
                "U": qU_mean,
                "V": qV_mean,
                "v0": qv0_mean,
            }

        else:
            ## MAP estimation

            U = tf.Variable(
                tf.random.normal(
                    [self.p, self.latent_dim], stddev=np.sqrt(1 / self.p)
                )
            )
            V = tf.Variable(
                tf.random.normal(
                    [self.latent_dim, self.q], stddev=np.sqrt(1 / self.q)
                )
            )
            v0 = tf.Variable(
                tf.random.normal(
                    [1, self.q], stddev=np.sqrt(1 / self.q)
                )
            )

            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

            losses = tfp.math.minimize(
                lambda: -target_log_prob_fn(U, V, v0),
                optimizer=optimizer,
                num_steps=n_iters,
                # convergence_criterion=convergence_criterion,
            )
            

            self.loss_trace = losses #[:n_iters]

            self.param_dict = {
                "U": U,
                "V": V,
                "v0": v0,
            }

        return


if __name__ == "__main__":

    n = 100
    p = 5
    q = 5
    k_true = 1

    n_repeats = 10

    for _ in range(n_repeats):
        X = np.random.uniform(low=-3, high=3, size=(n, p))
        U_true = np.random.normal(0, 0.1, size=(p, k_true))
        V_true = np.random.normal(0, 0.1, size=(k_true, q))

        linear_predictor = X @ U_true @ V_true
        size_factors = np.random.uniform(low=1e-1, high=2, size=n).reshape(-1, 1)
        linear_predictor += np.log(size_factors)
        Y_mean = np.exp(linear_predictor)
        Y = np.random.poisson(Y_mean)

        grrr = GRRR(latent_dim=k_true)
        grrr.fit(X=X, Y=Y, use_vi=False, size_factors=size_factors)

        U_est, V_est, v0_est = grrr.param_dict["U"].numpy(), grrr.param_dict["V"].numpy(), grrr.param_dict["v0"].numpy()
        preds = np.exp(X @ U_est @ V_est + v0_est + np.log(size_factors))

        plt.scatter(Y_mean[:, 0], preds[:, 0])
        plt.show()

        import ipdb; ipdb.set_trace()







