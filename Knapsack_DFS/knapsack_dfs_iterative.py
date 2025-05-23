import time
import random
class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

    def __repr__(self):
        return f"Item(poids={self.weight}, valeur={self.value})"


class KnapsackIterative:
    def __init__(self, items, capacity):
        self.items = items
        self.capacity = capacity
        self.best_value = 0  # suivis de la valeur 
        self.best_items = [] # suivis des items

    def solve(self):
        stack = [(0, 0, 0, [])]  # Pile pour l'iteration (index, poids actuel, valeur actuelle, items sélectionnés)
        
        while stack:
            index, current_weight, current_value, selected_items = stack.pop()
            
            # Si on a parcouru tous les items
            if index == len(self.items):
                if current_value > self.best_value:
                    self.best_value = current_value
                    self.best_items = selected_items.copy()
                continue

            # Cas où l'item courant est inclus (si le poids le permet)
            if current_weight + self.items[index].weight <= self.capacity:
                stack.append((
                    index + 1,
                    current_weight + self.items[index].weight,
                    current_value + self.items[index].value,
                    selected_items + [self.items[index]]
                ))
            
            # Cas où l'item courant est exclu
            stack.append((index + 1, current_weight, current_value, selected_items))
        
        return self.best_value, self.best_items


if __name__ == "__main__":
    # Generer des items aléatoires
    items = [Item(random.randint(1, 20), random.randint(1, 20)) for _ in range(20)]
    capacity = 100


    knapsack = KnapsackIterative(items, capacity)
    start_time = time.time()
    max_value, selected_items = knapsack.solve()
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Maximum value achievable: {max_value}")
    print("Items in the sack:")
    for item in selected_items:
        print(f"  - {item}")
    
    print(f"Execution time: {execution_time} seconds")