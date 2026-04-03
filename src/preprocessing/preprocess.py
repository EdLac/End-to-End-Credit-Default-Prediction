import pandas as pd
import numpy as np
import os

print("--- Lancement du script de preprocessing ---")

# Définition du chemin vers les données
data_path = "data/raw/Loan_Data.csv" 

if os.path.exists(data_path):
    print("Le fichier de données a bien été trouvé.")

    # Chargement des données
    df = pd.read_csv(data_path)
    print(f"Taille du dataset : {df.shape[0]} lignes et {df.shape[1]} colonnes.")
    
    # Afficher les 5 premières lignes
    print("\nVoici les premières lignes :")
    print(df.head())
else:
    print(f"Erreur : Impossible de trouver le fichier à ce chemin : {data_path}")