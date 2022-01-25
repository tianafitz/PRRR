import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import multivariate_normal
import torch
from torch import nn


class PoissonGLM(nn.Module):
    def __init__(self, X, Y, size_factors=None):
        super(PoissonGLM, self).__init__()
        self.n, self.p = X.shape
        _, self.q = Y.shape
        self.B = nn.Parameter(1e-1 * torch.randn([self.p, self.q]))
        if size_factors is None:
            self.log_size_factors = torch.log(Y.sum(1)).unsqueeze(1)
        else:
            self.log_size_factors = torch.log(size_factors)

    def forward(self, X):
        Y_mean = torch.exp(torch.mm(X, self.B) + self.log_size_factors)
        return Y_mean

    def loss(self, Y, pred_means):
        model_dist = torch.distributions.poisson.Poisson(rate=pred_means)
        LL = model_dist.log_prob(Y).sum()
        return -LL


if __name__ == "__main__":

    n = 1000
    p = 1
    q = 10

    n_repeats = 10
    B_estimated = []
    B_trues = []
    for _ in range(n_repeats):

        X = np.random.uniform(low=-1, high=1, size=(n, p))
        B_true = np.random.normal(loc=0, scale=1, size=(p, q))
        # Y_mean = np.exp(X @ B_true)
        linear_predictor = X @ B_true
        # size_factors = np.random.uniform(low=1, high=2, size=n).reshape(-1, 1)
        size_factors = np.random.choice(np.arange(50, 100), size=n, replace=True).reshape(-1, 1)
        linear_predictor += np.log(size_factors)
        Y_mean = np.exp(linear_predictor)

        Y = np.random.poisson(lam=Y_mean)
        # import ipdb; ipdb.set_trace()

        ## Remove samples with no counts
        nonzero_idx = np.where(Y.sum(1) != 0)[0]
        Y = Y[nonzero_idx]
        Y_mean = Y_mean[nonzero_idx]
        X = X[nonzero_idx]

        glm = PoissonGLM(X=torch.tensor(X).float(), Y=torch.tensor(Y).float(), size_factors=torch.tensor(size_factors).float())
        optimizer = torch.optim.Adam(glm.parameters(), lr=1e-2)
        epochs = 10000

        for epoch in range(epochs):
            optimizer.zero_grad()
            Y_mean_pred = glm(torch.tensor(X).float())

            # get loss for the predicted output
            loss = glm.loss(torch.tensor(Y).float(), Y_mean_pred)

            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print("epoch {}, loss {}".format(epoch, loss.item()))

        # preds = np.exp(X @ glm.B.detach().numpy() + np.log(Y.sum(1).reshape(-1, 1)))
        glm.eval()
        # glm.B = torch.nn.Parameter(torch.tensor(B_true).float())
        preds = glm.forward(torch.tensor(X).float()).detach().numpy()
        pred_samples = np.random.poisson(lam=preds)
        preds_rand = np.exp(
            X @ np.random.normal(size=(p, q)) + np.log(Y.sum(1).reshape(-1, 1))
        )
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(Y[:, 0], pred_samples[:, 0])
        plt.subplot(122)
        plt.scatter(Y[:, 0], preds_rand[:, 0])
        plt.show()

        plt.scatter(B_true.squeeze(), glm.B.detach().numpy())
        plt.show()
        import ipdb

        ipdb.set_trace()
