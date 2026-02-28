from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_performance(y_true, y_pred):
    """
    Returns basic classification metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }