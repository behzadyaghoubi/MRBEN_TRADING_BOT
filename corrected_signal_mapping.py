def correct_signal_mapping(predictions):
    """
    Correct signal mapping function
    predictions: numpy array of shape (n_samples, 3) with probabilities
    returns: list of signal strings
    """
    signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    predicted_classes = np.argmax(predictions, axis=1)
    return [signal_map[cls] for cls in predicted_classes]


def get_signal_confidence(predictions):
    """
    Get confidence for each prediction
    """
    return np.max(predictions, axis=1)
