import numpy as np

class SetCoveringProblem:
    def __init__(self, nom_fichier):
        self.nom_fichier = nom_fichier
        self.m = None  # Nombre de contraintes (lignes)
        self.n = None  # Nombre de colonnes
        self.couts = None  # Coûts des colonnes
        self.contraintes = None  # Liste des contraintes
        self.matrice_binaire = None  # Matrice binaire des contraintes

    def lire_fichier(self):
        """Lit le fichier et initialise les attributs."""
        with open(self.nom_fichier, 'r') as fichier:
            # Lire la première ligne
            self.m, self.n = map(int, fichier.readline().split())
            
            # Lire les coûts des colonnes
            self.couts = []
            while len(self.couts) < self.n:
                ligne = fichier.readline().strip()
                if not ligne:
                    continue  # Ignorer les lignes vides
                self.couts.extend(map(int, ligne.split()))
            
            # Lire les contraintes
            self.contraintes = []
            for _ in range(self.m):
                # Lire le nombre de colonnes qui couvrent la ligne
                while True:
                    ligne = fichier.readline().strip()
                    if not ligne:
                        continue  # Ignorer les lignes vides
                    nb_colonnes = int(ligne)
                    break
                
                # Lire les indices des colonnes
                colonnes = []
                while len(colonnes) < nb_colonnes:
                    ligne = fichier.readline().strip()
                    if not ligne:
                        continue  # Ignorer les lignes vides
                    colonnes.extend(map(int, ligne.split()))
                
                self.contraintes.append(colonnes)

    def creer_matrice_binaire(self):
        """Crée la matrice binaire des contraintes."""
        # Initialiser une matrice de zéros de taille (m, n)
        self.matrice_binaire = np.zeros((self.m, self.n), dtype=int)
        
        # Remplir la matrice
        for i, colonnes in enumerate(self.contraintes):
            for j in colonnes:
                self.matrice_binaire[i, j - 1] = 1  # Les indices commencent à 0 en Python

    def afficher_informations(self):
        """Affiche les informations du problème."""
        print(f"Nombre de lignes (m) : {self.m}")
        print(f"Nombre de colonnes (n) : {self.n}")
        print(f"Coûts des colonnes : {self.couts[:10]}...")  # Affiche les 10 premiers coûts
        print(f"Première contrainte dans la matrice : {self.matrice_binaire[0]}")
        print(f"Première contrainte dans la liste : {self.contraintes[0]}")

if __name__ == "__main__":
    scp = SetCoveringProblem('scp41.txt')
    scp.lire_fichier()
    scp.creer_matrice_binaire()
    scp.afficher_informations()