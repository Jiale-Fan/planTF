import torch


def sort_predictions(predictions, probability, k=6):
    """Sort the predictions based on the probability of each mode.
    Args:
        predictions (torch.Tensor): The predicted trajectories [b, k, t, 2].
        probability (torch.Tensor): The probability of each mode [b, k].
    Returns:
        torch.Tensor: The sorted predictions [b, k', t, 2].
    """
    indices = torch.argsort(probability, dim=-1, descending=True)
    assert indices.max().item() < predictions.size(1), "max indice " + str(indices.max().item()) + " exceeds the plan number of modes " + str(predictions.size(1))
    sorted_prob = probability[torch.arange(probability.size(0))[:, None], indices]
    sorted_predictions = predictions[
        torch.arange(predictions.size(0))[:, None], indices
    ]
    return sorted_predictions[:, :k], sorted_prob[:, :k]
