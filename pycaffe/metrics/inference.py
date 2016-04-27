import numpy as np

def metrics_from_net(net, metrics, key_label, key_score, threshold=0.5):
    """
    Run metrics on a given net
    """
    net.forward()
    y_true = net.blobs[key_label].data
    y_prob = net.blobs[key_score].data
    return metrics_from_blob(metrics, y_true, y_prob, threshold)

def metrics_from_blob(metrics, y_true, y_prob, threshold=0.5):
    """
    Run metrics on a given blob
    """
    # Normalize blob matrix to a 2D array
    y_true = np.squeeze(y_true)
    if y_true.ndim == 1:
        dim = y_true.shape[0]
        y_true = y_true.reshape((1, dim))
    # Generate predictions
    y_pred = np.array([[prob>=threshold for prob in preds] for preds in y_prob])
    return metrics(y_true, y_pred)
