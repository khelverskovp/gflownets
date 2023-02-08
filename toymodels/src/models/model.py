import torch
import torch.nn as nn

from utils import food_items

N_total = len(food_items)

class FlowModel(nn.Module):
  def __init__(self, num_hid):
    super().__init__()
    # encode the input for a dish as a into a list with zeroes and ones
    # N_total ingredients = N_total input neurons
    self.mlp = nn.Sequential(nn.Linear(N_total, num_hid), nn.LeakyReLU(),
                             # We also output N_total numbers, since there are up to
                             # 6 possible actions (and thus child states), but we 
                             # will mask those outputs for patches that are 
                             # already there.
                             nn.Linear(num_hid, N_total))
  
  def forward(self, x):
    # We take the exponential to get positive numbers, since flows must be positive,
    # and multiply by (1 - x) to give 0 flow to actions we know we can't take
    # (in this case, x[i] is 1 if a feature is already there, so we know we 
    # can't add it again).
    F = self.mlp(x).exp() * (1 - x)
    return F

class TBModel(nn.Module):
  def __init__(self, num_hid):
    super().__init__()
    # encode the input for a dish as a into a list with zeroes and ones
    # N_total ingredients = N_total input neurons
    self.mlp = nn.Sequential(nn.Linear(N_total, num_hid), nn.LeakyReLU(),
                             # We also output N_total numbers, since there are up to
                             # 6 possible actions (and thus child states), but we 
                             # will mask those outputs for patches that are 
                             # already there.
                             nn.Linear(num_hid, N_total*2))
    # log Z is just a single number                        
    self.logZ = nn.Parameter(torch.zeros(1))
  
  def forward(self, x):
    logits = self.mlp(x)
    # Slice the logits, and mask invalid actions (since we're predicting 
    # log-values), we use -100 since exp(-100) is tiny, but we don't want -inf)
    P_F = logits[..., :N_total] * (1 - x) + x * -100
    P_B = logits[..., N_total:] * x + (1 - x) * -100
    return P_F, P_B