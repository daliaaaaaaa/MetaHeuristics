class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

    def __repr__(self):
        return f"Item(poids={self.weight}, valeur={self.value})"


class KnapsackDFS:
    def __init__(self, items, capacity):
        self.items = items
        self.capacity = capacity
        self.best_value = 0  # suivis de la valeur 
        self.best_items = [] # suivis des items

    def solve(self):

        self._dfs(0, 0, 0, [])  # 1er appel de la fonction récursive
        return self.best_value, self.best_items

    def _dfs(self, index, current_weight, current_value,selected_items):
        
        #Fonction recursive

        # Si on a parcourus tous les items ou si la on atteint la capacité maximale
        if index == len(self.items):
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_items = selected_items.copy() 
            return

        # Exploration du cas où l'item courant est inclus
        if current_weight + self.items[index].weight <= self.capacity:
            selected_items.append(self.items[index])
            self._dfs( #appel recursif en incluant l'item courant
                index + 1,
                current_weight + self.items[index].weight,
                current_value + self.items[index].value,
                selected_items,
            )
            selected_items.pop()  #Exclure l'item courant

        # Exploration du cas où l'item courant est exclu
        self._dfs( #appel recursif en excluant l'item courant
            index + 1,
            current_weight,
            current_value,
            selected_items
        )

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

    knapsack = KnapsackDFS(items, capacity)
    max_value,selected_items = knapsack.solve()

    print(f"Maximum value achievable: {max_value}")
    print("Items in the sack:")
    for item in selected_items:
        print(f"  - {item}")