# import the model from toymodels\smileyFaces.py
import torch
import torch.nn as nn

# load the functions in dishes.py
from toymodels.dishes import *

class FlowModel(nn.Module):
    def __init__(self, num_hid):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(15, num_hid), nn.LeakyReLU(), nn.Linear(num_hid,15))
    
    def forward(self, x):

        F = self.mlp(x).exp() * (1 - x)
    
        return F

model = torch.load("models/dishes.pth")

print(model)

N_samples = 10

for i in range(N_samples):
    dish = []
    edge_flow_prediction = model(encodeIngredients(dish).float())

    for t in range(3):
        policy = edge_flow_prediction / edge_flow_prediction.sum()

        if t > 0:
            policy = torch.cat([policy, torch.tensor([0.1])])
        

        action = Categorical(policy).sample()

        if action.item() == 9:    
            break
        
        dish = dish + [breakfast[action.item()]] if action.item() < 3 else dish + [lunch[action.item() - 3]] if action.item() < 6 else dish + [dinner[action.item() - 6]]

        edge_flow_prediction = model(encodeIngredients(dish).float())
    
    reward = rewardFunction(dish)

    print(dish, reward)

print(model(encodeIngredients([])).sum())



