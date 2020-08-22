import torch
"""Args:
    features: list of feature torch.Tensor
    targets:  list of target torch.Tensor
   return block of features and targets
"""
def concat_feature_target(features, targets):
    return torch.cat(features,0), torch.cat(targets,0)