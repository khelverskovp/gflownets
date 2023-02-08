import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm
import matplotlib.pyplot as pp
import pdb

from utils import food_items, dish_reward, dish_parents, dish_to_tensor
from model import TBModel

import hydra

from datetime import datetime

import wandb

@hydra.main(config_path='conf/', config_name="default_config.yaml")
def main(cfg):
  wandb.init(project="toy-project", entity="gflownets")
  wandb.config.epochs = 50000
  wandb.config.batch_size = 4

  # Instantiate model and optimizer
  model = TBModel(100)
  opt = torch.optim.Adam(model.parameters(), 3e-4)

  N_total = len(food_items)

  # Let's keep track of the losses and the faces we sample
  tb_losses = []
  tb_sampled_dishes = []
  # To not complicate the code, I'll just accumulate losses here and take a 
  # gradient step every `update_freq` episode.
  minibatch_loss = 0
  update_freq = 2

  logZs = []
  for episode in tqdm.tqdm(range(50000), ncols=40):
    # Each episode starts with an "empty state"
    state = []
    # Predict P_F, P_B
    P_F_S,P_B_S = model(dish_to_tensor(state))
    total_P_F = 0
    total_P_B = 0
    for t in range(3):
      # Here P_F is logits, so we want the Categorial to compute the softmax for us
        cat = Categorical(logits=P_F_S)
        action = cat.sample()
        # "Go" to the next state
        new_state = state + [food_items[action]]
        # Accumulate the P_F sum
        total_P_F += cat.log_prob(action)

        if t == 2:
            # If we've built a complete dish, we're done, so the reward is > 0
            reward = torch.tensor(dish_reward(new_state)).float()
        
        # re recompute P_F and P_B for new_state
        P_F_S,P_B_S = model(dish_to_tensor(new_state))
        # Here we accumulate P_B going backwards from "new_state". We're also just 
        # going to use opposite semantics for the backward policy. I.e., for P_F action
        # "i" just added the ingredient "i" for P_B we'll assume action "i" removes
        # the ingredient "i". This way we can just keep the same indices.
        total_P_B += Categorical(logits=P_B_S).log_prob(action)

        # continue iterating
        state = new_state
    
    # we're done with the trajectory, compute its loss. Since the reward can sometimes be zero, 
    # instead of log(0) we'll clip the log-reward to -20.
    loss = (model.logZ + total_P_F - torch.log(reward).clip(-20) -total_P_B).pow(2)
    minibatch_loss += loss

    # add the dish to the list, and if we are at an
    # update episode, take a gradient step
    tb_sampled_dishes.append(state)
    if episode % update_freq == 0:
        tb_losses.append(minibatch_loss.item())
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0
        logZs.append(model.logZ.item())
    
    wandb.log({"loss": loss[-1]})
    wandb.watch(model)

  f, ax = pp.subplots(2, 1, figsize=(10,6))
  pp.sca(ax[0])
  pp.plot(tb_losses)
  pp.yscale('log')
  pp.ylabel('loss')
  pp.sca(ax[1])
  pp.plot(np.exp(logZs))
  pp.ylabel('estimated Z')

  pp.show()

  # save model and add the date to the name 
  torch.save(model, f"models/model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth")

if __name__ == "__main__":
  main()


    