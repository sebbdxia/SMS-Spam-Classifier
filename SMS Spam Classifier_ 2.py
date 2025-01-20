import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import streamlit as st

# Télécharger les ressources nécessaires pour NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Constantes
OUTPUT_DIR = "spam_classifier_output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "spam_classifier_model.pkl")
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nettoyage des données
def clean_text(text: str) -> str:
    """Nettoie le texte pour la classification."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)  # Remplace les URLs par 'url'
    text = re.sub(r'[^a-z\s]', '', text)  # Supprime les caractères non alphabétiques
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return ' '.join(
        lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words
    )

# Charger ou sauvegarder le modèle et le vectoriseur
def load_model_and_vectorizer():
    """Charge le modèle et le vectoriseur à partir des fichiers."""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            return model, vectorizer
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
    return None, None

def save_model_and_vectorizer(model, vectorizer):
    """Sauvegarde le modèle et le vectoriseur."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

# Entraîner le modèle
def train_model(data: pd.DataFrame):
    """Entraîne un modèle Random Forest avec un TfidfVectorizer."""
    vectorizer = TfidfVectorizer(max_features=5000)
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    messages = data['message'].astype(str).apply(clean_text)
    labels = data['label'].astype(int)
    X = vectorizer.fit_transform(messages)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Calculer les poids des échantillons
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # Optimisation des hyperparamètres
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Extraire les résultats du GridSearch
    results = pd.DataFrame(grid_search.cv_results_)

    # Visualisation des résultats du GridSearch
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=results['param_n_estimators'], y=results['mean_test_score'], label='Mean F1 Score')
    plt.title('Résultats de la Recherche de Grille')
    plt.xlabel("Nombre d'Estimateurs")
    plt.ylabel('Score F1 Moyen')
    plt.legend()
    grid_search_path = os.path.join(OUTPUT_DIR, 'grid_search_results.png')
    plt.savefig(grid_search_path)
    plt.close()

    model = grid_search.best_estimator_
    model.fit(X_train, y_train)
    save_model_and_vectorizer(model, vectorizer)

    # Évaluation sur les données de test
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (aire = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC (Receiver Operating Characteristic)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(OUTPUT_DIR, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()

    st.subheader("Évaluation sur les données de test")
    st.text(classification_report(y_test, y_pred))
    st.image(roc_path, caption="Courbe ROC")
    st.image(grid_search_path, caption="Résultats de la Recherche de Grille")

    return model, vectorizer

# Générer des métriques
def generate_metrics(data: pd.DataFrame, model, vectorizer):
    """Génère les métriques de performance et une matrice de confusion."""
    messages = data['message'].astype(str).apply(clean_text)
    X = vectorizer.transform(messages)
    y_true = data['label'].astype(int)
    y_pred = model.predict(X)

    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
    plt.title('Matrice de Confusion (Confusion Matrix)')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    confusion_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(confusion_path)
    plt.close()

    return report, confusion_path

# Interface utilisateur Streamlit
st.title("Analyseur SMS Spam - Classificateur avec Visualisation")

# Réentraîner le modèle
st.header("Réentraîner le Modèle")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file, delimiter='\t', header=None, names=['label', 'message'])
        data['label'] = data['label'].map({'ham': 0, 'spam': 1})
        model, vectorizer = train_model(data)
        st.success("Modèle réentraîné avec succès.")

        report, confusion_path = generate_metrics(data, model, vectorizer)
        st.subheader("Matrice de Confusion")
        st.image(confusion_path)
        st.subheader("Métriques de Performance")
        st.write(pd.DataFrame(report).transpose())
    except Exception as e:
        st.error(f"Erreur : {e}")

# Charger modèle et vectoriseur
model, vectorizer = load_model_and_vectorizer()

# Section de prédiction
st.header("Prédire un SMS")
user_input = st.text_area("Entrez un SMS à analyser :", height=100)

if st.button("Classer le SMS"):
    if not model or not vectorizer:
        st.warning("Modèle non disponible. Veuillez réentraîner le modèle.")
    else:
        try:
            input_vectorized = vectorizer.transform([clean_text(user_input)])
            prediction = model.predict(input_vectorized)[0]
            prediction_proba = model.predict_proba(input_vectorized)[0]
            st.success(f"Ce SMS est classé comme : **{'Spam' if prediction == 1 else 'Ham (Non-Spam)'}**")
            st.write(f"Probabilité de Spam : {prediction_proba[1]:.2f}")
            st.write(f"Probabilité de Ham : {prediction_proba[0]:.2f}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")