import numpy as np


def demographic_parity_difference(y_pred, sensitive_attr):
    """
    DPD = P(Y=1 | A=1) - P(Y=1 | A=0)
    """
    p1 = np.mean(y_pred[sensitive_attr == 1])
    p0 = np.mean(y_pred[sensitive_attr == 0])
    return p1 - p0


def equal_opportunity_difference(y_true, y_pred, sensitive_attr):
    """
    EOD = TPR(A=1) - TPR(A=0)
    """
    tpr1 = np.sum((y_pred == 1) & (y_true == 1) & (sensitive_attr == 1)) / max(
        np.sum((y_true == 1) & (sensitive_attr == 1)), 1
    )
    tpr0 = np.sum((y_pred == 1) & (y_true == 1) & (sensitive_attr == 0)) / max(
        np.sum((y_true == 1) & (sensitive_attr == 0)), 1
    )
    return tpr1 - tpr0