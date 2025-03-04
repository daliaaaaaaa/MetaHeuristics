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
    items = [
        Item(23, 92),
        Item(31, 57),
        Item(29, 49),
        Item(44, 68),
        Item(53, 60),
        Item(38, 43),
        Item(63, 67),
        Item(85, 84),
        Item(89, 87),
        Item(82, 72),
    ]
    capacity = 165

    knapsack = KnapsackIterative(items, capacity)
    max_value, selected_items = knapsack.solve()

    print(f"Maximum value achievable: {max_value}")
    print("Items in the sack:")
    for item in selected_items:
        print(f"  - {item}")
