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

        A = yield tfd.Gamma(
            concentration=tf.ones([self.p, self.latent_dim]),
            rate=tf.ones([self.p, self.latent_dim]),
            name="A",
        )

        B = yield tfd.Gamma(
            concentration=tf.ones([self.latent_dim, self.q]),
            rate=tf.ones([self.latent_dim, self.q]),
            name="B",
        )

        # Intercept (multiplicative in this case)
        b0 = yield tfd.Gamma(
            concentration=tf.ones([1, self.q]),
            rate=tf.ones([1, self.q]),
            name="b0",
        )

        predicted_rate = tf.multiply(tf.matmul(tf.matmul(X, A), B), b0)

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

        def target_log_prob_fn(A, B, b0):
            return model.log_prob((A, B, b0, Y))

        if use_vi:
            # ------- Specify variational families -----------

            # Variational parameter means

            qA_mean = tf.Variable(tf.random.normal([self.p, self.latent_dim]))
            qA_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([self.p, self.latent_dim]), bijector=tfb.Softplus()
            )

            qB_mean = tf.Variable(tf.random.normal([self.latent_dim, self.q]))
            qB_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([self.latent_dim, self.q]), bijector=tfb.Softplus()
            )

            qb0_mean = tf.Variable(tf.random.normal([1, self.q]))
            qb0_stddv = tfp.util.TransformedVariable(
                1e-4 * tf.ones([1, self.q]), bijector=tfb.Softplus()
            )

            def factored_normal_variational_model():
                qA = yield tfd.LogNormal(loc=qA_mean, scale=qA_stddv, name="qA")
                qB = yield tfd.LogNormal(loc=qB_mean, scale=qB_stddv, name="qB")
                qb0 = yield tfd.LogNormal(loc=qb0_mean, scale=qb0_stddv, name="qb0")

            # Surrogate posterior that we will try to make close to p
            surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
                factored_normal_variational_model
            )

            # --------- Fit variational inference model using MC samples and gradient descent ----------

            convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
                atol=1e-1
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

            A_mean = tf.exp(qA_mean + 0.5 * tf.square(qA_stddv))
            B_mean = tf.exp(qB_mean + 0.5 * tf.square(qB_stddv))
            b0_mean = tf.exp(qb0_mean + 0.5 * tf.square(qb0_stddv))

            self.param_dict = {
                "A": A_mean,
                "B": B_mean,
                "b0": b0_mean,
                "A_lognormal_mean": qA_mean,
                "A_lognormal_stddev": qA_stddv,
                "B_lognormal_mean": qB_mean,
                "B_lognormal_stddev": qB_stddv,
                "b0_lognormal_mean": qb0_mean,
                "b0_lognormal_stddev": qb0_stddv,
            }

        else:
            ## MAP estimation

            A = tfp.util.TransformedVariable(
                tf.random.gamma(alpha=1., shape=[self.p, self.latent_dim]),
                bijector=tfb.Softplus()
            )
            B = tfp.util.TransformedVariable(
                tf.random.gamma(alpha=1., shape=[self.latent_dim, self.q]),
                bijector=tfb.Softplus()
            )
            # b0 = tfp.util.TransformedVariable(
            #     tf.random.gamma(alpha=1., shape=[1, self.q]),
            #     bijector=tfb.Softplus()
            # )
            b0 = tfp.util.TransformedVariable(
                tf.ones(shape=[1, self.q]),
                bijector=tfb.Softplus()
            )

            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

            trace_fn = lambda traceable_quantities: {'loss': traceable_quantities.loss, 'A': A, 'B': B, 'b0': b0}
            losses = tfp.math.minimize(
                lambda: -target_log_prob_fn(A, B, b0),
                optimizer=optimizer,
                num_steps=n_iters,
                trace_fn=trace_fn,
            )
            

            self.loss_trace = losses #[:n_iters]

            self.param_dict = {
                "A": A,
                "B": B,
                "b0": b0,
            }

        return


if __name__ == "__main__":

    n = 1000
    p = 5
    q = 5
    k_true = 1

    n_repeats = 10

    for _ in range(n_repeats):
        X = np.random.uniform(low=0, high=3, size=(n, p))
        A_true = np.random.gamma(1., size=(p, k_true))
        B_true = np.random.gamma(1., size=(k_true, q))

        Y_mean = X @ A_true @ B_true
        Y = np.random.poisson(Y_mean)

        prrr = PRRR(latent_dim=k_true)
        size_factors = np.ones((n, 1))
        prrr.fit(X=X, Y=Y, use_vi=False, n_iters=5_000) #, size_factors=size_factors)

        A_est, B_est, b0_est = prrr.param_dict["A"].numpy(), prrr.param_dict["B"].numpy(), prrr.param_dict["b0"].numpy()
        preds = np.multiply(np.multiply(X @ A_est @ B_est, b0_est), prrr.size_factors)

        plt.scatter(Y_mean[:, 0], preds[:, 0])
        plt.show()
        import ipdb; ipdb.set_trace()
