import numpy as np
import os
import time
from math import ceil
class Benchmark:
    def __init__(self, file_path, benchmark_type):
        self.file_path = file_path
        self.benchmark_type = benchmark_type
        self.subsets, self.universe_size, self.num_subsets = self.read_benchmark()

    def read_benchmark(self):
        """Lecture d'un benchmark MCP à partir d'un fichier texte"""
        with open(self.file_path, "r") as file:
            lines = file.readlines()

        m, n = map(int, lines[0].split())
        cost_lines_to_skip = ceil(n / 12) if self.benchmark_type == "4" else ceil(n / 15)
        subset_start_index = 1 + cost_lines_to_skip  # Start of subset coverage data
        
        data_lines = lines[subset_start_index:]
        row_to_subsets = {}
        index = 0

        for row in range(1, m + 1):
            num_subsets = int(data_lines[index].strip())
            index += 1

            subsets = []
            while len(subsets) < num_subsets:
                subsets.extend(map(int, data_lines[index].split()))
                index += 1

            row_to_subsets[row] = subsets

        subset_to_rows = {}
        for row, subsets in row_to_subsets.items():
            for subset in subsets:
                if subset not in subset_to_rows:
                    subset_to_rows[subset] = []
                subset_to_rows[subset].append(row)

        return subset_to_rows, m, n
class PSO_MCP:
    def __init__(self, benchmark, num_particles=100, num_iterations=50, c1=1.5, c2=2, w=1, vmax=10, early_stop=30):
        self.benchmark = benchmark
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.vmax = vmax
        self.early_stop = early_stop
        self.k = ceil(benchmark.num_subsets * 2 / 3)  # Define k

    def fitness_func(self, solution):
        """Évalue une solution en mesurant le nombre d'éléments couverts"""
        covered_elements = set()
        for idx, selected in enumerate(solution):
            if selected and idx in self.benchmark.subsets:
                covered_elements.update(self.benchmark.subsets[idx])
        return len(covered_elements), len(covered_elements) / self.benchmark.universe_size

    def run(self):
        """Exécute l'optimisation PSO"""
        np.random.seed(42)
        particles = np.zeros((self.num_particles, self.benchmark.num_subsets), dtype=int)
        
        for i in range(self.num_particles):
            ones_indices = np.random.choice(self.benchmark.num_subsets, self.k, replace=False)
            particles[i, ones_indices] = 1
        
        velocities = np.zeros((self.num_particles, self.benchmark.num_subsets))
        personal_best = particles.copy()
        personal_best_scores = np.array([self.fitness_func(p)[1] for p in particles])
        global_best_idx = np.argmax(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()
        global_best_covered, global_best_score = self.fitness_func(global_best)
        
        best_scores = []
        no_improvement = 0
        start_time = time.time()

        for iteration in range(self.num_iterations):
            for i in range(self.num_particles):
                velocities[i] = self.update_velocity_binary(velocities[i], particles[i], personal_best[i], global_best)
                particles[i] = self.binary_update(particles[i], velocities[i])
                
                num_covered, score = self.fitness_func(particles[i])
                
                if score > personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = score
                
                if score > global_best_score:
                    global_best = particles[i].copy()
                    global_best_covered, global_best_score = num_covered, score
                    no_improvement = 0  # Reset early stop counter

            best_scores.append(global_best_score)
            print(f"Iteration {iteration+1}/{self.num_iterations} - Covered: {global_best_covered}, Score: {global_best_score * 100:.4f} %")

            if no_improvement >= self.early_stop:
                print(f"Arrêt anticipé après {iteration+1} itérations.")
                break
            
            no_improvement += 1

        execution_time = time.time() - start_time
        print(f"Optimisation terminée en {execution_time:.2f} sec - Meilleure couverture: {global_best_covered}, Score final: {global_best_score *100 :.4f} %")
        
        return global_best, global_best_covered, global_best_score, execution_time

    def update_velocity_binary(self, velocity, particle, personal_best, global_best):
        """Binary PSO velocity update using discrete difference"""
        r1, r2 = np.random.rand(len(particle)), np.random.rand(len(particle))
        velocity = (self.w * velocity +
                    self.c1 * r1 * (personal_best != particle).astype(int) +
                    self.c2 * r2 * (global_best != particle).astype(int))
        return np.clip(velocity, -self.vmax, self.vmax)

    def enforce_k_constraint(self, particle):
        """S'assure que le vecteur binaire a exactement k éléments activés"""
        if np.sum(particle) != self.k:
            ones = np.where(particle == 1)[0]
            zeros = np.where(particle == 0)[0]

            if len(ones) > self.k:
                np.random.shuffle(ones)
                particle[ones[self.k:]] = 0  # Désactive les surplus
            elif len(ones) < self.k:
                np.random.shuffle(zeros)
                particle[zeros[:(self.k - len(ones))]] = 1  # Active des zéros pour atteindre k
        return particle
    

    def binary_update(self, particle, velocity):
        """Mise à jour des particules en utilisant la transformation sigmoïde"""
        prob = 1 / (1 + np.exp(-velocity))
        updated_particle = (np.random.rand(*particle.shape) < prob).astype(int)
        return self.enforce_k_constraint(updated_particle)


def experiment(benchmark_files, benchmark_folder, type):
    """Exécute l'optimisation PSO sur un benchmark"""

    results = []

    for file in benchmark_files:
        file_path = os.path.join(benchmark_folder, file)
        benchmark = Benchmark(file_path, type)
        if type == "4":
            # Iterations: 50, W: 0.5, c1: 2.5, c2: 2.5, Vmax: 6, Particles: 100
            pso = PSO_MCP(benchmark, num_particles=100, num_iterations=50, c1=2.5, c2=2.5, w=0.5, vmax=6)
        elif type == "A":
            # Iterations: 50, W: 0.4, c1: 1, c2: 1.0, Vmax: 6, Particles: 50
            pso = PSO_MCP(benchmark, num_particles=50, num_iterations=50, c1=1, c2=1.0, w=0.4, vmax=6)
        elif type == "B":
            pso = PSO_MCP(benchmark, num_particles=50, c2=1, w=0.5)
        elif type == "C":
            # Iterations: 50, W: 0.4, c1: 1, c2: 1.0, Vmax: 6, Particles: 50
            pso = PSO_MCP(benchmark, num_particles=50, num_iterations=50, c1=1, c2=1.0, w=0.4, vmax=6)
        else:
            print(f"Type de benchmark inconnu: {type}")

        print(f"\n Exécution de PSO sur {file}...")
        best_solution, best_covered, best_score, exec_time = pso.run()
        results.append((file, best_solution, best_covered, best_score, exec_time))
    return results


def run_all_benchmarks():
    """Exécute l'optimisation PSO sur tous les benchmarks"""
    benchmark_folders = ["Benchmark/4", "Benchmark/A", "Benchmark/B", "Benchmark/C"]
    all_results = []

    for benchmark_folder in benchmark_folders:
        print(f"\nDossier de benchmarks: {benchmark_folder}")
        benchmark_files = sorted([f for f in os.listdir(benchmark_folder) if f.endswith(".txt")])
        results = experiment(benchmark_files, benchmark_folder, benchmark_folder[-1])
        
        for file, _, best_cov, score, time in results:  # Fix tuple unpacking
            all_results.append((benchmark_folder, file, best_cov, score, time))

    print("\nRésumé des résultats après toutes les exécutions :")
    for folder, file, best_cov, score, time in all_results:
        print(f"{folder}/{file}: Couvert = {best_cov}, Score = {score*100:.4f}%, Temps = {time:.2f}s")

def run_one_benchmark(benchmark_folder):
    """Exécute l'optimisation PSO sur un benchmark spécifique"""
    benchmark_files = sorted([f for f in os.listdir(benchmark_folder) if f.endswith(".txt")])
    results = experiment(benchmark_files, benchmark_folder, benchmark_folder[-1])
    
    print("\nRésumé des résultats après toutes les exécutions :")
    for file, _, best_cov, score, time in results:  # Remove 'folder' (incorrect variable)
        print(f"{file}: Couvert = {best_cov}, Score = {score:.4f}, Temps = {time:.2f}s")

def run_one_file(path):
    benchmark = Benchmark(path, path.split("Benchmark/")[1][0])
    pso = PSO_MCP(benchmark)
    print(f"\nExécution de PSO sur {path}...")
    best_solution, best_covered, best_score, exec_time = pso.run()
    print(f"\nRésumé des résultats :")
    print(f"{path}: Couvert = {best_covered}, Score = {best_score:.4f}, Temps = {exec_time:.2f}s")

if __name__ == "__main__":
    print("Choisissez une option:")
    print("1. Exécuter un seul fichier")
    print("2. Exécuter un seul benchmark")
    print("3. Exécuter tous les benchmarks")

    choix = input("Entrez votre choix (1, 2 ou 3): ")

    if choix == "1":
        path = input("Entrez le chemin du fichier benchmark: ")
        run_one_file(path)
    elif choix == "2":
        benchmark_folder = input("Entrez le dossier du benchmark (ex: Benchmark/4): ")
        run_one_benchmark(benchmark_folder)
    elif choix == "3":
        run_all_benchmarks()
    else:
        print("Option invalide. Veuillez choisir 1, 2 ou 3.")
