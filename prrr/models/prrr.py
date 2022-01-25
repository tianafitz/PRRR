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

NUM_VI_ITERS = 100000
LEARNING_RATE_VI = 0.05


class PRRR:
    def __init__(self, n_latent_dim):
        self.n_latent_dim = n_latent_dim

    def prrr_model(self, X):

        A = yield tfd.Gamma(
            concentration=tf.fill([self.p, self.n_latent_dim], 2.0),
            rate=tf.ones([self.p, self.n_latent_dim]),
            name="A",
        )

        B = yield tfd.Gamma(
            concentration=tf.fill([self.n_latent_dim, self.q], 2.0),
            rate=tf.ones([self.n_latent_dim, self.q]),
            name="B",
        )

        count_mean = tf.matmul(tf.matmul(X.astype("float32"), A), B)

        if self.size_factors is not None:
            count_mean = tf.multiply(count_mean, self.size_factors)

        Y = yield tfd.Poisson(rate=count_mean, name="Y")

    def fit(self, X, Y, size_factors=None):

        assert X.shape[0] == Y.shape[0]
        self.n, self.p = X.shape
        _, self.q = Y.shape
        self.size_factors = size_factors

        rrr_model = functools.partial(self.prrr_model, X=X)

        model = tfd.JointDistributionCoroutineAutoBatched(rrr_model)

        def target_log_prob_fn(A, B):
            return model.log_prob((A, B, Y))

        # ------- Specify variational families -----------

        # Variational parameter means

        qA_mean = tf.Variable(tf.random.normal([self.p, self.n_latent_dim]))
        qA_stddv = tfp.util.TransformedVariable(
            1e-4 * tf.ones([self.p, self.n_latent_dim]), bijector=tfb.Softplus()
        )

        qB_mean = tf.Variable(tf.random.normal([self.n_latent_dim, self.q]))
        qB_stddv = tfp.util.TransformedVariable(
            1e-4 * tf.ones([self.n_latent_dim, self.q]), bijector=tfb.Softplus()
        )

        def factored_normal_variational_model():
            qA = yield tfd.LogNormal(loc=qA_mean, scale=qA_stddv, name="qA")
            qB = yield tfd.LogNormal(loc=qB_mean, scale=qB_stddv, name="qB")

        # Surrogate posterior that we will try to make close to p
        surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
            factored_normal_variational_model
        )

        # --------- Fit variational inference model using MC samples and gradient descent ----------

        convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
            atol=1e-1
        )
        optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE_VI)

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

        return


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

            prrr = PRRR(n_latent_dim=k_rrr)
            prrr.fit(X=X, Y=Y)
            plt.plot(prrr.loss_trace)
            plt.show()
