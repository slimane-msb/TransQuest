import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

# Compute Pearson correlation coefficient
def pearsonr_metric (x, y):
    r, _ = pearsonr(y, x)
    return ("Pearson correlation coefficient: %.3f" % r)

# Compute mean absolute error
def mae_metric (x, y):
    mae = np.mean(np.abs(y - x))
    return ("Mean absolute error: %.3f" % mae)

# Compute root mean squared error
def rmse_metric (x, y):
    rmse = np.sqrt(np.mean((y - x)**2))
    return ("Root mean squared error: %.3f" % rmse)


