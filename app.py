import streamlit as st
import pandas as pd
import joblib

# Chargement du modèle et du scaler
model = joblib.load(r"C:\Users\Pc\Desktop\2A\2A\DS\projet2\extra_trees_model.pkl")
scaler = joblib.load(r"C:\Users\Pc\Desktop\2A\2A\DS\projet2\scaler.pkl")


st.title("Détection d'attaques IoT")
st.write("Application de classification des types d’attaques réseau IoT")

# Upload du fichier CSV
uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)

    st.subheader("Aperçu des données")
    st.dataframe(df_new.head())

    # Suppression colonne inutile si présente
    if "Unnamed: 0" in df_new.columns:
        df_new = df_new.drop(columns=["Unnamed: 0"])

    # Normalisation
    df_scaled = scaler.transform(df_new)

    # Prédiction
    predictions = model.predict(df_scaled)

    df_new["Prediction_Attack_type"] = predictions

    st.subheader("Résultats de la prédiction")
    st.dataframe(df_new.head())

    # Téléchargement
    csv = df_new.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger les résultats",
        csv,
        "predictions.csv",
        "text/csv"
    )
