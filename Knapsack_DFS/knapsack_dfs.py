class Item:
    # Representation of the item
    def __init__(self, weight, value):
        self.weight = weight  # Weight of the item
        self.value = value    # Value of the item

    def __repr__(self):
        return f"Item(weight={self.weight}, value={self.value})"


class KnapsackDFS:
    # Solving the Knapsack Problem using Depth First Search (DFS).
    def __init__(self, items, capacity):
        # initialize the items and capacity
        self.items = items
        self.capacity = capacity
        self.best_value = 0  # Track the best value found
        self.best_items = [] # Track the items that give the best value

    def solve(self):
        """
        Solve the Knapsack Problem using DFS.
        return the maximum value achievable without exceeding the capacity.
        """
        self._dfs(0, 0, 0, [])  # Start DFS from the first item
        return self.best_value, self.best_items

    def _dfs(self, index, current_weight, current_value,selected_items):
        """
        Recursive DFS function that explores all possible combinations of items.
         index: Index of the current item being considered.
         current_weight: Total weight of the current selection.
         current_value: Total value of the current selection.
         selected_items: List of items selected so far.
        """
        # Base case: If all items are considered or capacity is reached
        if index == len(self.items):
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_items = selected_items.copy() # Update the best items
            return

        # Exploring the option of including the current item
        if current_weight + self.items[index].weight <= self.capacity:
            selected_items.append(self.items[index])  # Include the item
            self._dfs(  # Recur with the item included
                index + 1,
                current_weight + self.items[index].weight,
                current_value + self.items[index].value,
                selected_items,
            )
            selected_items.pop()  # Backtrack (remove the item)

        # Exploring the option of excluding the current item
        self._dfs( # Recur without the item included
            index + 1,
            current_weight,
            current_value,
            selected_items
        )


# Example usage
if __name__ == "__main__":
    # Defining items and capacity
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

    # Solution
    knapsack = KnapsackDFS(items, capacity)
    max_value,selected_items = knapsack.solve()

    # Display
    print(f"Maximum value achievable: {max_value}")
    print("Items in the sack:")
    for item in selected_items:
        print(f"  - {item}")