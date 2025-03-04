import random
import matplotlib.pyplot as plt

class SCPGeneticAlgorithm:
    def __init__(self, filename, pop_size=500, max_generations=10000, k=7, stagnation_limit=300):
        """Initialize the Genetic Algorithm with enhanced parameters."""
        self.filename = filename
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.k = k  
        self.stagnation_limit = stagnation_limit
        self.n, self.m, self.subsets = self.load_data()
        self.population = self.generate_population()
        self.best_fitness_per_generation = []
        self.best_solution = None
        self.best_fitness = 0
        self.stagnation_counter = 0

    def load_data(self):
        """Loads the SCP dataset from a file."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        n, m = map(int, lines[0].split())
        subsets = []
        index = 1
        while index < len(lines):
            if " " in lines[index]:
                subsets.append(set(map(int, lines[index].split())))
                index += 1
            else:
                subset_size = int(lines[index].strip())
                subsets.append(set(map(int, lines[index + 1].split())))
                index += 2
        return n, m, subsets

    def fitness(self, solution):
        """Calculates the fitness of a given solution."""
        covered = set()
        for i in solution:
            if i < len(self.subsets):
                covered.update(self.subsets[i])
        return len(covered)

    def generate_population(self):
        """Generates the initial population."""
        return [random.sample(range(self.m), self.k) for _ in range(self.pop_size)]

    def selection(self):
        """Tournament Selection (Avoids premature convergence)."""
        tournament = random.sample(self.population, k=5)
        return max(tournament, key=self.fitness)

    def crossover(self, parent1, parent2):
        return [random.choice(gene) if random.random() > 0.1 else random.randint(0, self.m - 1) for gene in zip(parent1, parent2)]

    def mutate(self, solution):
        mutation_rate = 0.05 if self.stagnation_counter < 20 else 0.3

        if random.random() < mutation_rate:
            index = random.randint(0, len(solution) - 1)
            new_value = random.randint(0, self.m - 1)
            while new_value in solution:
                new_value = random.randint(0, self.m - 1)
            solution[index] = new_value
        return solution

    def run(self):

        for gen in range(self.max_generations):
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1, parent2 = self.selection(), self.selection()
                child1, child2 = self.crossover(parent1, parent2), self.crossover(parent2, parent1)
                child1, child2 = self.mutate(child1), self.mutate(child2)
                new_population.extend([child1, child2])
            
            # Elitism: Preserve top solutions
            elite_count = max(1, self.pop_size // 20)
            self.population = sorted(new_population, key=self.fitness, reverse=True)[:self.pop_size - elite_count]
            self.population += sorted(self.population, key=self.fitness, reverse=True)[:elite_count]
            
            current_best_fitness = self.fitness(self.population[0])
            self.best_fitness_per_generation.append(current_best_fitness)
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[0]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            if gen % 50 == 0:
                print(f"Generation {gen}: Best Fitness = {self.best_fitness}")
            
            if self.stagnation_counter >= self.stagnation_limit:
                print(f"\nStopped early at generation {gen} due to stagnation.")
                break
        
        print("\nBest Solution:", self.best_solution)
        print("Covered Elements:", self.best_fitness)
        self.plot_results()
        return self.best_solution

    def plot_results(self):
        """Plots the evolution of the best fitness."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_fitness_per_generation, marker='o', linestyle='-', color='b', label="Best Fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness (Covered Elements)")
        plt.title("Evolution of Best Fitness Over Generations")
        plt.legend()
        plt.grid(True)
        plt.show()

# Run the optimized SCP Genetic Algorithm
if __name__ == "__main__":
    # ga = SCPGeneticAlgorithm("scp41.txt", pop_size=500, max_generations=10000, k=7, stagnation_limit=300)
    ga = SCPGeneticAlgorithm("scp41.txt", pop_size=200, max_generations=1000, k=7, stagnation_limit=1000)

    ga.run()
