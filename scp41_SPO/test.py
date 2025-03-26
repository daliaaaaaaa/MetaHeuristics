from file import SetCoveringProblem
import numpy as np

def evaluate(solution, costs, cover_matrix):
    """Calcule le coût total et le nombre de contraintes couvertes."""
    total_cost = sum(solution * costs)
    covered_constraints = np.any(cover_matrix[:, solution == 1], axis=1)
    num_covered = np.sum(covered_constraints)
    
    return num_covered, total_cost


def pso_mcp(filename, num_particles=30, num_iterations=100, w=0.5, c1=2.0, c2=2.0):
    """Applique PSO pour résoudre SCP et affiche les meilleures fitness par itération."""
    scp = SetCoveringProblem(filename)
    scp.lire_fichier()
    scp.creer_matrice_binaire()

    num_elements = scp.m
    num_subsets = scp.n
    costs = np.array(scp.couts)
    cover_matrix = scp.matrice_binaire

    # Initialisation des particules et vitesses
    particles = np.random.randint(2, size=(num_particles, num_subsets))
    velocities = np.random.uniform(-1, 1, (num_particles, num_subsets))

    # Initialisation des meilleures solutions
    personal_best = particles.copy()
    personal_best_scores = [evaluate(p, costs, cover_matrix) for p in particles]
    global_best = personal_best[np.argmax([score[0] for score in personal_best_scores])]
    global_best_score = max(personal_best_scores, key=lambda x: x[0])

    fitness_evolution = []

    for iteration in range(num_iterations):
        for i in range(num_particles):
            inertia = w * velocities[i]
            cognitive = c1 * np.random.rand(num_subsets) * (personal_best[i] - particles[i])
            social = c2 * np.random.rand(num_subsets) * (global_best - particles[i])
            velocities[i] = inertia + cognitive + social
            particles[i] = np.where(np.random.rand(num_subsets) < 1 / (1 + np.exp(-velocities[i])), 1, 0)

            new_score = evaluate(particles[i], costs, cover_matrix)
            if new_score[0] > personal_best_scores[i][0]:
                personal_best[i] = particles[i]
                personal_best_scores[i] = new_score
            if new_score[0] > global_best_score[0]:
                global_best = particles[i]
                global_best_score = new_score

        fitness_evolution.append(global_best_score[0])
        print(f"Itération {iteration + 1}: Meilleure fitness = {global_best_score[0]}, Coût = {global_best_score[1]}")

    return global_best, global_best_score, fitness_evolution, cover_matrix


if __name__ == "__main__":
    best_solution, best_score, fitness_evolution, final_matrix = pso_mcp("scp41.txt")

    print("\nMeilleure solution trouvée :", best_solution)
    print("Score de la meilleure solution :", best_score)
    print("\nMatrice de couverture finale :\n", final_matrix)

    with open("fitness_evolution.txt", "w") as f:
        for i, fitness in enumerate(fitness_evolution):
            f.write(f"Itération {i+1}: {fitness}\n")

    print("\nL'évolution de la fitness a été sauvegardée dans 'fitness_evolution.txt'.")
