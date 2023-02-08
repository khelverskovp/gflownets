from utils import food_items, dish_reward, dish_to_tensor
from torch.distributions.categorical import Categorical
import torch
import numpy as np

model = torch.load("models/model.pth")

# sample some dishes

N_samples = 10

for i in range(N_samples):
    dish = []
    for t in range(2):
      # The policy is just normalizing, and gives us the probability of each action
      edge_flow_prediction = model(dish_to_tensor(dish))
      if i == 0 and t == 0:
        print(edge_flow_prediction.sum().item())
        print(edge_flow_prediction)
      policy = edge_flow_prediction / edge_flow_prediction.sum()
      # Sample the action
      action = Categorical(probs=policy).sample() 
      # "Go" to the next state
      dish = dish + [food_items[action]]
    
    reward = dish_reward(dish)

    print(f"Sample {i+1}: {dish} with reward {reward}")

total = 0

for food in food_items:
  edge_flow_prediction = model(dish_to_tensor([food]))
  total += edge_flow_prediction.sum().item()

print(total)



