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