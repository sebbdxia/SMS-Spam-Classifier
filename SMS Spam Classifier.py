import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
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
nltk.download('omw-1.4')

# Constantes
MODEL_PATH = "spam_classifier_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
OUTPUT_DIR = r'C:\Users\sbond\Desktop\Nouveau dossier'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger ou sauvegarder le modèle et le vectoriseur
def load_model_and_vectorizer():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Modèle ou vectoriseur non trouvé. Entraînez le modèle pour générer ces fichiers.")
        st.stop()

def save_model_and_vectorizer(model, vectorizer):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

# Nettoyage des données
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Entraînement du modèle
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

# Ajout d'un menu dans la barre latérale
menu = st.sidebar.selectbox("Menu", ["Prédire un SMS", "Réentraîner le Modèle"])

if menu == "Prédire un SMS":
    # Section de prédiction
    st.header("Prédire un SMS")
    user_input = st.text_area("Entrez un SMS à analyser :", height=100)

    model, vectorizer = load_model_and_vectorizer()
    if st.button("Classer le SMS"):
        if user_input.strip() == "":
            st.warning("Veuillez entrer un SMS valide.")
        else:
            input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(input_vectorized)[0]
            label = "Spam" if prediction == 1 else "Ham (Non-Spam)"
            st.success(f"Ce SMS est classé comme : **{label}**")

elif menu == "Réentraîner le Modèle":
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
                model, vectorizer = train_model(data)
                report, confusion_path = generate_metrics(data, model, vectorizer)

                st.success("Le modèle a été réentraîné et les métriques ont été générées.")

                # Afficher les visualisations
                st.header("Résultats de l'Évaluation")
                st.subheader("Matrice de Confusion")
                st.image(confusion_path, caption="Matrice de Confusion")

                st.subheader("Métriques de Performance")
                st.write(pd.DataFrame(report).transpose())
            else:
                st.error("Les labels doivent être 'ham' ou 'spam'.")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")