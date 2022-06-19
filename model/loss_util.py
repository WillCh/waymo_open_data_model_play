import torch

def combined_regression_classification_loss(
    gt: torch.Tensor, pred: torch.Tensor, likelihood: torch.Tensor,
    regression_weight: float, 
    classification_weight:float) -> torch.Tensor:
    """Loss function which is linear combination of regression and
        classification. We only compute the regression loss for the
        cloest prediction based on L2 distance.
    
    Args:
        gt: groundtruth tensor, whose dimensions are [B, timestamps, x, y].
        pred: prediction tensor, whose dimensions are [B, num trajs,
            timestamps, x, y].
        likelihood: predicted likelihood, whose dimensions are 
            [B, num trajs].
        regression_weight: float weight for regression loss.
        classification_weight: float weight for classification loss.

    Returns:
        1D tensor whose dimensions are [B], representing the final loss.
    """
    