import random
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm

import matplotlib.pyplot as plt
import numpy as np

breakfast = ["eggs", "bacon", "pancakes"]
lunch = ["sandwich", "soup", "salad"]
dinner = ["steak", "chicken", "pasta"]

#make a list of all the ingredients
ingredientsKeys = breakfast + lunch + dinner

# make a reward function that returns the number of items that are from the same list as the reward
def rewardFunction(ingredients):
    # make a list of the number of items that are in each list
    numBreakfast = len([item for item in ingredients if item in breakfast])
    numLunch = len([item for item in ingredients if item in lunch])
    numDinner = len([item for item in ingredients if item in dinner])

    # get the highest, second highest and lowest number of items in a list
    maxNum = max(numBreakfast, numLunch, numDinner)
    minNum = min(numBreakfast, numLunch, numDinner)

    # get the number in between
    if numBreakfast != maxNum and numBreakfast != minNum:
        midNum = numBreakfast
    elif numLunch != maxNum and numLunch != minNum:
        midNum = numLunch
    elif numDinner != maxNum and numDinner != minNum:
        midNum = numDinner
    else:
        midNum = 0
    
    reward = 1 * maxNum - 0.5 * midNum - 0.25 * minNum

    # if the ingredients contain more than 1 of the same item, make the reward 0
    if len(ingredients) != len(set(ingredients)):
        reward = 0
    
    # return the number of items that are in the same list as the most items
    return reward

# encode ingredients as a tensor
def encodeIngredients(ingredients):
    
    return torch.tensor([i in ingredients for i in breakfast + lunch + dinner], dtype=torch.float)

class FlowModel(nn.Module):
    def __init__(self, num_hid):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(9, num_hid), nn.LeakyReLU(), nn.Linear(num_hid,9))
    
    def forward(self, x):

        F = self.mlp(x).exp() * (1 - x)
    
        return F
    
def dish_parents(state):
    parents_states = []
    parents_actions = []

    for ingredient in state:
        parents_states.append([i for i in state if i != ingredient])
        parents_actions.append(ingredientsKeys.index(ingredient))
    
    return parents_states, parents_actions

if __name__ == "__main__":
    # instantiate model and optimizer 
    model = FlowModel(100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # keep track of losses and the dishes we sample
    losses = []
    dishes = []

    minibatch_loss = 0
    update_freq = 4
    for episode in tqdm.tqdm(range(50000), ncols=40):
        state = []

        edge_flow_prediction = model(encodeIngredients(state).float())

        for t in range(5):
            policy = edge_flow_prediction / edge_flow_prediction.sum()

            # add a termial state that can be chosen at any time if the state is not empty
            if t > 0:
                policy = torch.cat([policy, torch.tensor([0.1])])

            action = Categorical(policy).sample()

            # if the action is the terminal state, we are done
            if action.item() == 9:
                reward = rewardFunction(new_state)

                edge_flow_prediction = torch.zeros(9)
                break
            
            new_state = state + [breakfast[action.item()]] if action.item() < 3 else state + [lunch[action.item() - 3]] if action.item() < 6 else state + [dinner[action.item() - 6]]

            # now we compute the loss, first enumerate the parents
            parents_states, parents_actions = dish_parents(new_state)
            
            # and compute the edge flows F(s,a) for each parent
            px = torch.stack([encodeIngredients(p) for p in parents_states]).float()
            pa = torch.tensor(parents_actions, dtype=torch.long)
            parent_edge_flow_preds = model(px)[torch.arange(len(parents_states)), pa]

            # now we need to compute the reward and F(s,a) for the new state
            if t == 4:
                reward = rewardFunction(new_state)

                edge_flow_prediction = torch.zeros(9)
            else:
                reward = 0
                edge_flow_prediction = model(encodeIngredients(new_state).float())
            
            # the loss as per the equation in the paper
            flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
            minibatch_loss += flow_mismatch

            state = new_state

        dishes.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            minibatch_loss = 0

    plt.figure(figsize=(10,5))
    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()

    # save the model to a file
    torch.save(model, "toymodels/models/dishes.pth")


