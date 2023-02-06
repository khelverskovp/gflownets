from utils import dish_reward, food_items
from math import factorial
from itertools import combinations

""" for i in range(1,6):
    comps = factorial(15) / (factorial(i) * factorial(15-i))
    total_reward += comps
    print(f"Number of dishes with {i} ingredients: {comps}") """
    

total_reward = 0

for r in range(1,3):
    for comp in combinations(range(len(food_items)),r):
        dish = []
        for idx in comp:
            dish.append(food_items[idx])
        total_reward += dish_reward(dish)

print(total_reward)
