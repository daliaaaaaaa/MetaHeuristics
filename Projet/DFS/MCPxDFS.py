from math import ceil
import time

class Benchmark:
    def __init__(self, file_path, benchmark_type):
        self.file_path = file_path
        self.benchmark_type = benchmark_type
        self.subsets, self.universe_size, self.num_subsets = self.read_benchmark()
    
    def read_benchmark(self):
        """Lecture d'un benchmark MCP à partir d'un fichier texte"""
        with open(self.file_path, "r") as file:
            lines = file.readlines()
        #Lecture de la 1er ligne m et n
        m, n = map(int, lines[0].split())
        #Lecture des lignes de coûts a sauter 
        cost_lines_to_skip = ceil(n / 12) if self.benchmark_type == "4" else ceil(n / 15)
        subset_start_index = 1 + cost_lines_to_skip  # Start of subset coverage data
        
        data_lines = lines[subset_start_index:]
        row_to_subsets = {}
        index = 0
        
        #generate dictionary of subsets for each row
        for row in range(1, m + 1):
            num_subsets = int(data_lines[index].strip())
            index += 1
            subsets = []
            while len(subsets) < num_subsets:
                subsets.extend(map(int, data_lines[index].split()))
                index += 1
            row_to_subsets[row] = subsets
        #Inverse
        #generate dictionary of rows for each subset
        subset_to_rows = {}
        for row, subsets in row_to_subsets.items():
            for subset in subsets:
                if subset not in subset_to_rows:
                    subset_to_rows[subset] = []
                subset_to_rows[subset].append(row)
        return subset_to_rows, m, n
    def print_binary_matrix(self):
        """Affiche la matrice binaire représentant les relations élément-sous-ensemble."""
        matrix = [[0] * self.num_subsets for _ in range(self.universe_size)]
        for subset, elements in self.subsets.items():
            for element in elements:
                matrix[element - 1][subset - 1] = 1  # Ajustement des index
        print("\nBinary Matrix Representation:")
        print("    " + " ".join(f"S{sub+1}" for sub in range(self.num_subsets)))  # En-tête
        for i, row in enumerate(matrix):
            print(f"E{i+1:2} " + " ".join(str(val) for val in row))

class DFSSolver:
    def __init__(self, benchmark, k, timeout=60):
        self.subsets = benchmark.subsets
        self.universe_size = benchmark.universe_size
        self.num_subsets = benchmark.num_subsets
        # self.k = ceil(benchmark.universe_size * 0.2) if benchmark.benchmark_type == "4" else ceil(benchmark.universe_size * 0.13)
        self.k = ceil(benchmark.universe_size * 2/3) 
        self.timeout = timeout
        self.start_time = None
        self.best_solution = []
        self.best_coverage = 0
        self.best_covered_elements = set()
        self.nodes_explored = 0
        self.timeout_occurred = False
    
    def dfs(self, current_subset_idx, remaining, selected_subsets, covered_elements):
        if self.is_timeout():
            self.timeout_occurred = True
            print("Timeout reached. Stopping exploration.")
            return
            
        self.nodes_explored += 1
        print(f"Exploring node {self.nodes_explored}: Covered {len(covered_elements)} elements")
        
        # If we've selected the required number of subsets
        if len(selected_subsets) == self.k:
            coverage = len(covered_elements)
            if coverage > self.best_coverage:
                self.best_coverage = coverage
                self.best_solution = selected_subsets.copy()
                self.best_covered_elements = covered_elements.copy()
                print(f"New best solution found! Coverage: {self.best_coverage}")
            return
        
        # Prune branch if we can't reach the required number of subsets
        if len(selected_subsets) + len(remaining) < self.k:
            print("Pruning branch: Not enough subsets left to reach k.")
            return
        
        all_remaining_elements = set()
        for subset in remaining:
            all_remaining_elements.update(set(self.subsets[subset]) - covered_elements)
        
        upper_bound = len(covered_elements) + len(all_remaining_elements)
        if upper_bound <= self.best_coverage:
            print("Pruning branch: Upper bound not better than current best.")
            return
        
        # Directly loop through all remaining subsets without any heuristic-based sorting
        for subset in remaining:
            if subset in remaining:
                print(f"Selecting subset {subset}")
                new_covered = covered_elements.union(set(self.subsets[subset]))
                selected_subsets.append(subset)
                remaining.remove(subset)
                
                self.dfs(subset, remaining, selected_subsets, new_covered)
                
                if self.timeout_occurred:
                    return
                
                print(f"Backtracking from subset {subset}")
                selected_subsets.pop()
                remaining.add(subset)
    
    def is_timeout(self):
        return time.time() - self.start_time > self.timeout
    
    def solve(self):
        self.start_time = time.time()
        remaining = set(range(1, self.num_subsets + 1))
        self.dfs(0, remaining, [], set())
        return {
            "selected_subsets": self.best_solution,
            "coverage": self.best_coverage,
            "nodes_explored": self.nodes_explored,
            "timeout_occurred": self.timeout_occurred,
            "time_taken": time.time() - self.start_time 
        }
    
    def print_solution_matrix(self):
        """Affiche la matrice binaire de la meilleure solution trouvée."""
        if not self.best_solution:
            print("Aucune solution trouvée.")
            return
        matrix = [[0] * len(self.best_solution) for _ in range(self.universe_size)]
        subset_indices = {subset: idx for idx, subset in enumerate(self.best_solution)}
        for subset in self.best_solution:
            for element in self.subsets[subset]:
                matrix[element - 1][subset_indices[subset]] = 1  # Ajustement des index
        
        print("\nBinary Matrix of Best Solution:")
        print("    " + " ".join(f"S{sub}" for sub in self.best_solution))  # En-tête
        for i, row in enumerate(matrix):
            print(f"E{i+1:2} " + " ".join(str(val) for val in row))


# Example usage
if __name__ == "__main__":
    import time
    
    # Example with a small benchmark file
    benchmark_file = "../Benchmark/A/scpa1.txt"  # Replace with your benchmark file
    benchmark_type = "A"  # Set according to your benchmark type
    timeout = 60  # Timeout in seconds (1 minute)
    
    # Read benchmark

    benchmark = Benchmark(benchmark_file, benchmark_type)
    print(f"Benchmark loaded: {benchmark.universe_size} elements, {benchmark.num_subsets} subsets")

    
    # Solve using DFS
    solver = DFSSolver(benchmark,timeout)
    result = solver.solve()
    
    # Print results
    print(f"Best solution: {result['selected_subsets']}")
    print(f"Coverage: {result['coverage']} out of {benchmark.universe_size} elements ({result['coverage']/benchmark.universe_size*100:.2f}%)")
    print(f"Nodes explored: {result['nodes_explored']}")
    print(f"Time taken: {result['time_taken']:.2f} seconds")
    # benchmark.print_binary_matrix()
    solver.print_solution_matrix()

    if result['timeout_occurred']:
        print(f"Timeout occurred after {timeout} seconds - returned best solution found so far")



