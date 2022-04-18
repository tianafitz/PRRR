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

class PRRR:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def prrr_model(self, X):

        U = yield tfd.Gamma(
            concentration=tf.ones([self.p, self.latent_dim]),
            rate=tf.ones([self.p, self.latent_dim]),
            name="U",
        )

        V = yield tfd.Gamma(
            concentration=tf.ones([self.latent_dim, self.q]),
            rate=tf.ones([self.latent_dim, self.q]),
            name="V",
        )

        # Intercept (multiplicative in this case)
        v0 = yield tfd.Gamma(
            concentration=tf.ones([1, self.q]),
            rate=tf.ones([1, self.q]),
            name="v0",
        )
        v0 = tf.ones([1, self.q])

        predicted_rate = tf.multiply(tf.matmul(tf.matmul(X, U), V), v0)

        if self.size_factors is not None:
            predicted_rate = tf.multiply(predicted_rate, self.size_factors)

        Y = yield tfd.Poisson(rate=predicted_rate, name="Y")

    def fit(self, X, Y, size_factors=None, use_total_counts_as_size_factors=True, use_vi=True, n_iters=2_000, learning_rate=1e-2):

        assert X.shape[0] == Y.shape[0]
        self.n, self.p = X.shape
        _, self.q = Y.shape
        if use_total_counts_as_size_factors and size_factors is None:
            size_factors = np.sum(Y, axis=1).astype(float).reshape(-1, 1)
        self.size_factors = size_factors
        X = X.astype("float32")

        rrr_model = functools.partial(self.prrr_model, X=X)

        model = tfd.JointDistributionCoroutineAutoBatched(rrr_model)

        def target_log_prob_fn(U, V, v0):
            return model.log_prob((U, V, v0, Y))

        if use_vi:
            # ------- Specify variational families -----------

            # Variational parameter means

            qU_mean = tf.Variable(tf.random.normal([self.p, self.latent_dim]))
            qU_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([self.p, self.latent_dim]), bijector=tfb.Softplus()
            )

            qV_mean = tf.Variable(tf.random.normal([self.latent_dim, self.q]))
            qV_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([self.latent_dim, self.q]), bijector=tfb.Softplus()
            )

            qv0_mean = tf.Variable(tf.random.normal([1, self.q]))
            qv0_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([1, self.q]), bijector=tfb.Softplus()
            )

            def factored_normal_variational_model():
                qU = yield tfd.LogNormal(loc=qU_mean, scale=qU_stddv, name="qA")
                qV = yield tfd.LogNormal(loc=qV_mean, scale=qV_stddv, name="qB")
                qv0 = yield tfd.LogNormal(loc=qv0_mean, scale=qv0_stddv, name="qb0")

            # Surrogate posterior that we will try to make close to p
            surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
                factored_normal_variational_model
            )

            # --------- Fit variational inference model using MC samples and gradient descent ----------

            # convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
            #     atol=1e-1
            # )
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

            losses = tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn,
                surrogate_posterior=surrogate_posterior,
                optimizer=optimizer,
                num_steps=n_iters,
                # convergence_criterion=convergence_criterion,
            )
            n_iters = optimizer._iterations.numpy()

            self.loss_trace = losses[:n_iters]

            U_mean = tf.exp(qU_mean + 0.5 * tf.square(qU_stddv))
            V_mean = tf.exp(qV_mean + 0.5 * tf.square(qV_stddv))
            v0_mean = tf.exp(qv0_mean + 0.5 * tf.square(qv0_stddv))

            self.param_dict = {
                "U": U_mean,
                "V": V_mean,
                "v0": v0_mean,
                "U_lognormal_mean": qU_mean,
                "U_lognormal_stddev": qU_stddv,
                "V_lognormal_mean": qV_mean,
                "V_lognormal_stddev": qV_stddv,
                "v0_lognormal_mean": qv0_mean,
                "v0_lognormal_stddev": qv0_stddv,
            }

        else:
            ## MAP estimation

            U = tfp.util.TransformedVariable(
                tf.random.gamma(alpha=1., shape=[self.p, self.latent_dim]),
                bijector=tfb.Softplus()
            )
            V = tfp.util.TransformedVariable(
                tf.random.gamma(alpha=1., shape=[self.latent_dim, self.q]),
                bijector=tfb.Softplus()
            )
            # b0 = tfp.util.TransformedVariable(
            #     tf.random.gamma(alpha=1., shape=[1, self.q]),
            #     bijector=tfb.Softplus()
            # )
            v0 = tfp.util.TransformedVariable(
                tf.ones(shape=[1, self.q]),
                bijector=tfb.Softplus()
            )

            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

            trace_fn = lambda traceable_quantities: {'loss': traceable_quantities.loss, 'U': U, 'V': V, 'v0': v0}
            losses = tfp.math.minimize(
                lambda: -target_log_prob_fn(U, V, v0),
                optimizer=optimizer,
                num_steps=n_iters,
                trace_fn=trace_fn,
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
        X = np.random.uniform(low=0, high=3, size=(n, p))
        U_true = np.random.gamma(1., size=(p, k_true))
        V_true = np.random.gamma(1., size=(k_true, q))

        Y_mean = X @ U_true @ V_true
        Y = np.random.poisson(Y_mean)

        prrr = PRRR(latent_dim=k_true)
        size_factors = np.ones((n, 1))
        prrr.fit(X=X, Y=Y, use_vi=True, n_iters=5_000) #, size_factors=size_factors)

        U_est, V_est, v0_est = prrr.param_dict["U"].numpy(), prrr.param_dict["V"].numpy(), prrr.param_dict["v0"].numpy()
        preds = np.multiply(np.multiply(X @ U_est @ V_est, v0_est), prrr.size_factors)

        plt.scatter(Y_mean[:, 0], preds[:, 0])
        plt.show()
        import ipdb; ipdb.set_trace()
