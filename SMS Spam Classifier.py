import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import streamlit as st

# Télécharger les ressources nécessaires
nltk.download('stopwords')
nltk.download('wordnet')

# Constantes
OUTPUT_DIR = r"C:\Users\sbond\Desktop\spam_classifier_project"
MODEL_PATH = os.path.join(OUTPUT_DIR, "spam_classifier_model.pkl")
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger ou sauvegarder le modèle et le vectoriseur
def load_model_and_vectorizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        validate_model_and_vectorizer(model, vectorizer)
        return model, vectorizer
    else:
        st.warning("Modèle ou vectoriseur introuvable. Veuillez fournir un fichier de données pour réentraîner.")
        return None, None

def save_model_and_vectorizer(model, vectorizer):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

# Validation du modèle et du vectoriseur
def validate_model_and_vectorizer(model, vectorizer, sample_text="test"):
    try:
        vectorizer.transform([sample_text])  # Vérifiez que le vectoriseur fonctionne
        model.predict(vectorizer.transform([sample_text]))  # Vérifiez que le modèle accepte les caractéristiques
    except ValueError:
        st.error("Erreur : le modèle et le vectoriseur ne sont pas compatibles. Réentraînez-les ensemble.")
        st.stop()

# Nettoyage des données
def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Entraîner le modèle
def train_model(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    model = MultinomialNB()
    messages = data['message'].astype(str).apply(clean_text)
    labels = data['label'].astype(int)

    X = vectorizer.fit_transform(messages)
    model.fit(X, labels)

    save_model_and_vectorizer(model, vectorizer)
    return model, vectorizer

# Analyse et visualisation
def generate_metrics(data, model, vectorizer):
    messages = data['message'].astype(str).apply(clean_text)
    X = vectorizer.transform(messages)
    y_true = data['label'].astype(int)
    y_pred = model.predict(X)
    
    # Rapport de classification
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    confusion_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(confusion_path)
    plt.close()

    return report, confusion_path

# Interface utilisateur Streamlit
st.title("SMS Spam Classifier avec Analyse et Visualisation")

# Charger le modèle et le vectoriseur existants
model, vectorizer = load_model_and_vectorizer()

# Section de prédiction
st.header("Prédire un SMS")
user_input = st.text_area("Entrez un SMS à analyser :", height=100)

if st.button("Classer le SMS"):
    if not model or not vectorizer:
        st.warning("Modèle non disponible. Réentraînez le modèle pour continuer.")
    elif user_input.strip() == "":
        st.warning("Veuillez entrer un SMS valide.")
    else:
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)[0]
        label = "Spam" if prediction == 1 else "Ham (Non-Spam)"
        st.success(f"Ce SMS est classé comme : **{label}**")

# Section pour réentraîner le modèle
st.header("Réentraîner le Modèle")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file, delimiter='\t', header=None, names=['label', 'message'])
        st.write("Fichier chargé avec succès.")
        st.write(data.head())

        if set(data['label'].unique()).issubset({'ham', 'spam'}):
            data['label'] = data['label'].map({'ham': 0, 'spam': 1})

            # Réentraîner le modèle
            model, vectorizer = train_model(data)
            st.success("Le modèle a été réentraîné avec succès.")

            # Génération des métriques
            report, confusion_path = generate_metrics(data, model, vectorizer)

            st.header("Résultats de l'Évaluation")
            st.subheader("Matrice de Confusion")
            st.image(confusion_path, caption="Matrice de Confusion")

            st.subheader("Métriques de Performance")
            st.write(pd.DataFrame(report).transpose())
        else:
            st.error("Les labels doivent être 'ham' ou 'spam'.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
