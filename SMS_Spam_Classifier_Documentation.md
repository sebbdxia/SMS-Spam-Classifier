# Documentation du Code : SMS Spam Classifier

Ce document explique en détail le fonctionnement du code pour la classification des SMS en spam ou non-spam (« Ham »). 

## Bibliothèques Utilisées
- os : Gestion des fichiers et des chemins.
- pandas : Manipulation de données sous forme de tableaux.
- joblib : Sauvegarde et chargement de modèles ou d'objets Python.
- sklearn : Contient des outils pour l’apprentissage automatique (modèles, métriques, etc.).
- matplotlib et seaborn : Visualisation des données (graphiques et matrice de confusion).
- nltk : Traitement du langage naturel (stopwords, lemmatisation).
- streamlit : Interface utilisateur pour les applications web interactives.

Structure Principale
1. **Constantes et Configuration
- MODEL_PATH : Chemin pour sauvegarder le modèle formé.
- VECTORIZER_PATH : Chemin pour sauvegarder le vectoriseur TF-IDF.
- OUTPUT_DIR : Dossier pour sauvegarder les résultats (ex. matrice de confusion).
- Le dossier de sortie est créé automatiquement si inexistant avec `os.makedirs()`.

2. **Préparation des Ressources NLTK
- Télécharge les stopwords, la base de lemmatisation et des ressources additionnelles via `nltk.download()`.

3. Fonctions Principales
a. load_model_and_vectorizer()
- Charge le modèle et le vectoriseur précédemment sauvegardés.
- Si les fichiers sont introuvables, affiche un message d’erreur et arrête l’exécution.

b. save_model_and_vectorizer()
- Sauvegarde le modèle et le vectoriseur dans des fichiers pour une utilisation ultérieure.

c. clean_text(text)
- Nettoie et prétraite un texte :
  - Convertit en minuscules.
  - Supprime les caractères non alphabétiques.
  - Retire les mots vides (« stopwords »).
  - Applique une lemmatisation sur chaque mot.

d. train_model(data)
- Entraîne un modèle de classification Naïve Bayes multinomial.
- Étapes :
  1. Nettoie les messages avec `clean_text()`.
  2. Vectorise les messages avec TF-IDF.
  3. Entraîne le modèle avec les labels donnés.
  4. Sauvegarde le modèle et le vectoriseur.

e. generate_metrics(data, model, vectorizer)
- Génère des métriques d’évaluation et des visualisations :
  - **Rapport de classification** : Précision, rappel, F1-score, etc.
  - **Matrice de confusion** : Visualisation des prédictions correctes et erronées.
  - Sauvegarde la matrice de confusion sous forme d’image.

4. Interface Utilisateur avec Streamlit
a. Menu Principal
- « Prédire un SMS » :
  - Permet à l’utilisateur d’entrer un texte et d’obtenir une classification (Spam ou Ham).
- « Réentraîner le Modèle » :
  - Charge un fichier CSV contenant les données (étiquettes « spam » et « ham »).
  - Réentraîne le modèle et affiche les métriques d’évaluation.

b. Prédire un SMS
1. Charge le modèle et le vectoriseur.
2. Transforme le texte entré en vecteur TF-IDF.
3. Prédit si le texte est « Spam » ou « Ham ».
4. Affiche le résultat à l’utilisateur.

c. Réentraîner le Modèle
1. Charge un fichier CSV avec des colonnes « label » et « message ».
2. Mappe les labels :
   - « ham » → 0.
   - « spam » → 1.
3. Entraîne le modèle avec les nouvelles données.
4. Génère les métriques et visualisations.

5. Métriques et Visualisations
- Rapport de classification :
  - Mesure la performance du modèle pour chaque classe (« spam » et « ham »).
- Matrice de confusion :
  - Représente les prédictions correctes et erronées sous forme de tableau.
  - Sauvegardée en tant qu’image dans le dossier de sortie.

Instructions pour Utilisation
1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Lancer l’application Streamlit :
   ```bash
   streamlit run SMS_Spam_Classifier.py
   ```
3. Interagir via l’interface utilisateur pour prédire ou réentraîner le modèle.

Améliorations Possibles
- Ajouter d’autres modèles de classification (SVM, Random Forest).
- Permettre le téléchargement direct des résultats.
- Améliorer le nettoyage du texte pour inclure des expressions régulières avancées.
- Ajouter une option pour évaluer le modèle sur un ensemble de test différent.

Conclusion

Ce projet offre une solution simple et interactive pour la classification des SMS en spam ou non-spam. Avec un pipeline clair et modulable, il peut être étendu pour inclure d'autres types de données ou méthodes d'apprentissage automatique.
