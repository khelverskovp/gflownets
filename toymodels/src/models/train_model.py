import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm
import matplotlib.pyplot as plt
import pdb

from utils import food_items, dish_reward, dish_parents, dish_to_tensor
from model import FlowModel

import datetime

import wandb

wandb.init(project="toy-project", entity="gflownets")

# Instantiate model and optimizer
F_sa = FlowModel(100)
opt = torch.optim.Adam(F_sa.parameters(), 0.001)

N_total = len(food_items)

if __name__ == "__main__":
  # Let's keep track of the losses and the faces we sample
  losses = []
  sampled_dishes = []
  # To not complicate the code, I'll just accumulate losses here and take a 
  # gradient step every `update_freq` episode.
  minibatch_loss = 0
  update_freq = 4

  wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100
}
  for episode in tqdm.tqdm(range(50000), ncols=40):
    # Each episode starts with an "empty state"
    state = []
    # Predict F(s, a)
    edge_flow_prediction = F_sa(dish_to_tensor(state))
    for t in range(2):
      # The policy is just normalizing, and gives us the probability of each action
      policy = edge_flow_prediction / edge_flow_prediction.sum()
      # Sample the action
      action = Categorical(probs=policy).sample() 
      # "Go" to the next state
      new_state = state + [food_items[action]]

      # Now we want to compute the loss, we'll first enumerate the parents
      parent_states, parent_actions = dish_parents(new_state)
      # And compute the edge flows F(s, a) of each parent
      px = torch.stack([dish_to_tensor(p) for p in parent_states])
      pa = torch.tensor(parent_actions).long()
      parent_edge_flow_preds = F_sa(px)[torch.arange(len(parent_states)), pa]
      # Now we need to compute the reward and F(s, a) of the current state,
      # which is currently `new_state`
      if t == 1: 
        # If we've built a complete face, we're done, so the reward is > 0
        # (unless the face is invalid)
        reward = dish_reward(new_state)
        # and since there are no children to this state F(s,a) = 0 \forall a
        edge_flow_prediction = torch.zeros(N_total)
      else:
        # Otherwise we keep going, and compute F(s, a)
        reward = 0
        edge_flow_prediction = F_sa(dish_to_tensor(new_state))

      # The loss as per the equation above
      flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
      minibatch_loss += flow_mismatch  # Accumulate
      # Continue iterating
      state = new_state

    # We're done with the episode, add the face to the list, and if we are at an
    # update episode, take a gradient step.
    sampled_dishes.append(state)
    if episode % update_freq == 0:
      losses.append(minibatch_loss.item())
      minibatch_loss.backward()
      opt.step()
      opt.zero_grad()
      minibatch_loss = 0
    
    wandb.log({"loss": losses[-1]})
    wandb.watch(F_sa)

  plt.figure(figsize=(10,3))
  plt.plot(losses)
  #plt.yscale('log')
  plt.show()

  

  # save model and add the date to the name 
  torch.save(F_sa, f"models/model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth")


    