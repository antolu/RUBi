def compute_acc(labels, predictions):
    """
    predictions: output of a forward pass through the model
    computes the accuracy by comparing them with the correct labels
    """
    output = (predictions > 0.5).float()
    correct = (predictions == labels).float().sum()
    acc = correct/output.shape[0]
    return acc

