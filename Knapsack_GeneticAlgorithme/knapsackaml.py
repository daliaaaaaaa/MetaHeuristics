import random
import matplotlib.pyplot as plt

# Définition de la classe Item
class Item:
    def __init__(self, name, value, weight):
        self.name = name
        self.value = value
        self.weight = weight

    def __repr__(self):
        return f"{self.name} (Valeur: {self.value}, Poids: {self.weight})"

# Définition de la classe Knapsack (sac à dos)
class Knapsack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def add_item(self, item):
        if self.total_weight() + item.weight <= self.capacity:
            self.items.append(item)
            return True
        return False

    def total_value(self):
        return sum(item.value for item in self.items)

    def total_weight(self):
        return sum(item.weight for item in self.items)

    def __repr__(self):
        return f"Knapsack (Valeur Totale: {self.total_value()}, Poids Total: {self.total_weight()})"

# Génération d'une liste fixe de 25 objets
def generate_fixed_items():
    return [
        Item("Item1", 10, 5), Item("Item2", 40, 10), Item("Item3", 30, 7), Item("Item4", 50, 15),
        Item("Item5", 35, 9), Item("Item6", 25, 6), Item("Item7", 45, 12), Item("Item8", 20, 4),
        Item("Item9", 55, 17), Item("Item10", 60, 20), Item("Item11", 70, 25), Item("Item12", 15, 3),
        Item("Item13", 80, 30), Item("Item14", 90, 35), Item("Item15", 95, 40), Item("Item16", 33, 8),
        Item("Item17", 22, 5), Item("Item18", 44, 11), Item("Item19", 66, 19), Item("Item20", 77, 21),
        Item("Item21", 88, 28), Item("Item22", 99, 33), Item("Item23", 100, 38), Item("Item24", 110, 45),
        Item("Item25", 120, 50)
    ]

# Algorithme Glouton (Greedy)
def greedy_knapsack(items, capacity):
    items = sorted(items, key=lambda x: x.value / x.weight, reverse=True)
    knapsack = Knapsack(capacity)
    for item in items:
        knapsack.add_item(item)
    return knapsack

# Algorithme Aléatoire
def random_knapsack(items, capacity):
    random.shuffle(items)
    knapsack = Knapsack(capacity)
    for item in items:
        knapsack.add_item(item)
    return knapsack

# Algorithme de Programmation Dynamique
def dynamic_knapsack(items, capacity):
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if items[i-1].weight <= w:
                dp[i][w] = max(items[i-1].value + dp[i-1][w-items[i-1].weight], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    
    knapsack = Knapsack(capacity)
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            knapsack.add_item(items[i-1])
            w -= items[i-1].weight

    return knapsack

# Algorithme Génétique
def genetic_knapsack(items, capacity, population_size=10, generations=100, mutation_rate=0.1):
    population = [[random.choice([0, 1]) for _ in range(len(items))] for _ in range(population_size)]

    def fitness(chromosome):
        total_value = sum(items[i].value for i in range(len(items)) if chromosome[i] == 1)
        total_weight = sum(items[i].weight for i in range(len(items)) if chromosome[i] == 1)
        return total_value if total_weight <= capacity else 0

    for _ in range(generations):
        population = sorted(population, key=fitness, reverse=True)
        next_generation = population[:2]

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(population[:5], 2)
            crossover_point = random.randint(1, len(items) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]

            if random.random() < mutation_rate:
                mutate_index = random.randint(0, len(items) - 1)
                child[mutate_index] = 1 - child[mutate_index]

            next_generation.append(child)

        population = next_generation

    best_solution = max(population, key=fitness)
    selected_items = [items[i] for i in range(len(items)) if best_solution[i] == 1]

    knapsack = Knapsack(capacity)
    for item in selected_items:
        knapsack.add_item(item)

    return knapsack

# Exécution et comparaison
capacity = 100
items = generate_fixed_items()

knapsack_greedy = greedy_knapsack(items, capacity)
knapsack_random = random_knapsack(items, capacity)
knapsack_dynamic = dynamic_knapsack(items, capacity)
knapsack_genetic = genetic_knapsack(items, capacity)

# Affichage des résultats
print("\n### Comparaison des Algorithmes ###")
print("Sac Glouton  :", knapsack_greedy)
print("Sac Aléatoire:", knapsack_random)
print("Sac Dynamique:", knapsack_dynamic)
print("Sac Génétique:", knapsack_genetic)

# Graphique comparatif
labels = ["Glouton", "Aléatoire", "Dynamique", "Génétique"]
values = [knapsack_greedy.total_value(), knapsack_random.total_value(), knapsack_dynamic.total_value(), knapsack_genetic.total_value()]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel("Méthodes")
plt.ylabel("Valeur totale du sac")
plt.title("Comparaison des méthodes du sac à dos")
plt.show()
