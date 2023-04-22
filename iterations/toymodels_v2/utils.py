import numpy as np
import torch

# breakfast products
# breakfast = ["eggs", "bacon", "bread", "yogurt", "cheese", "fruit", "pancake", "oats", "milk", "honey"]
breakfast = ["eggs", "bacon"]

# lunch = ["baguette", "ryebread", "cucumber", "salami", "mayo", "mustard", "pickles", "ham", "iceberg", "tomato"]
lunch = ["baguette", "ryebread"]

# dinner = ["chicken", "rice", "curry", "potatoes", "salt", "pepper", "pork", "sriracha", "vegetables", "chickpeas"]
dinner = ["chicken", "rice"]

# combine everything into one list
food_items = [*breakfast,*lunch,*dinner]

def get_category(food):
    if food in breakfast:
        return "breakfast"
    elif food in lunch:
        return "lunch"
    elif food in dinner:
        return "dinner"
    
    return "invalid food item"

# make dictionary for food items
food_category = {food : get_category(food) for food in food_items}

def is_valid(dish):
    # cannot have more ingredients than is available
    if len(dish) > len(breakfast) + len(lunch) + len(dinner):
        return False
    
    # cannot contain the same ingredient twice
    if len(np.unique(dish)) < len(dish):
        return False
    
    return True

def dish_reward(dish):
    if not is_valid(dish):
        return 0
    
    cats = np.array([food_category[food] for food in dish])

    n_breakfast = sum(cats == "breakfast") 
    n_lunch = sum(cats == "lunch") 
    n_dinner = sum(cats == "dinner") 

    counts = np.sort([n_breakfast,n_lunch,n_dinner])[::-1]

    return 1 * counts[0] - 0.5 * counts[1] - 0.25 * counts[2]

# encode dish to tensor
def dish_to_tensor(dish):
  return torch.tensor([i in dish for i in food_items]).float()

def dish_parents(state):
  parent_states = []  # states that are parents of state
  parent_actions = []  # actions that lead from those parents to state
  for ingredient in state:
    # For each dish part, there is a parent without that part
    parent_states.append([i for i in state if i != ingredient])
    # The action to get there is the corresponding index of that face part
    parent_actions.append(food_items.index(ingredient))
  return parent_states, parent_actions