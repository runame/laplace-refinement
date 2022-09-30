import numpy as np
from scipy.spatial.distance import pdist
from sklearn import metrics


def mmd_rbf(X, Y):
    """
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2)).
    Taken from: https://github.com/jindongwang/transferlearning

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    # Median heuristic --- use some hold-out samples
    all_samples = np.concatenate([X[:50], Y[:50]], 0)
    pdists = pdist(all_samples)
    sigma = np.median(pdists)
    gamma=1/(sigma**2)

    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()


def accuracy(y_pred, y_true):
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        return np.mean(y_pred.argmax(1) == y_true).mean()*100


def nll(y_pred, y_true):
    """
    Mean Categorical negative log-likelihood. `y_pred` is a probability vector.
    """
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        return metrics.log_loss(y_true, y_pred)


def brier(y_pred, y_true):
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        def one_hot(targets, nb_classes):
            res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
            return res.reshape(list(targets.shape)+[nb_classes])

        return metrics.mean_squared_error(y_pred, one_hot(y_true, y_pred.shape[-1]))


def calibration(pys, y_true, M=100):
    try:
        pys, y_true = pys.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        # Put the confidence into M bins
        _, bins = np.histogram(pys, M, range=(0, 1))

        labels = pys.argmax(1)
        confs = np.max(pys, axis=1)
        conf_idxs = np.digitize(confs, bins)

        # Accuracy and avg. confidence per bin
        accs_bin = []
        confs_bin = []
        nitems_bin = []

        for i in range(M):
            labels_i = labels[conf_idxs == i]
            y_true_i = y_true[conf_idxs == i]
            confs_i = confs[conf_idxs == i]

            acc = np.nan_to_num(np.mean(labels_i == y_true_i), 0)
            conf = np.nan_to_num(np.mean(confs_i), 0)

            accs_bin.append(acc)
            confs_bin.append(conf)
            nitems_bin.append(len(labels_i))

        accs_bin, confs_bin = np.array(accs_bin), np.array(confs_bin)
        nitems_bin = np.array(nitems_bin)

        ECE = np.average(np.abs(confs_bin-accs_bin), weights=nitems_bin/nitems_bin.sum())
        MCE = np.max(np.abs(accs_bin - confs_bin))

        # In percent
        ECE, MCE = ECE*100, MCE*100

        return ECE, MCE
