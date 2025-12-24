import streamlit as st
import pandas as pd
import joblib
import os

# Chemin du dossier courant (où se trouve app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Chargement du modèle et du scaler
model_path = os.path.join(current_dir, "extra_trees_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("Détection d'attaques IoT")
st.write("Application de classification des types d'attaques réseau IoT")

# Upload du fichier CSV
uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    
    st.subheader("Aperçu des données")
    st.dataframe(df_new.head())
    
    # Supprimer colonne inutile si présente
    if "Unnamed: 0" in df_new.columns:
        df_new = df_new.drop(columns=["Unnamed: 0"])
    
    try:
        # Récupérer les colonnes attendues par le scaler
        colonnes_attendues = scaler.feature_names_in_
        
        # Vérifier si toutes les colonnes nécessaires sont présentes
        colonnes_manquantes = set(colonnes_attendues) - set(df_new.columns)
        colonnes_en_trop = set(df_new.columns) - set(colonnes_attendues)
        
        if colonnes_manquantes:
            st.error(f" Colonnes manquantes dans le fichier : {list(colonnes_manquantes)}")
            st.stop()
        
        if colonnes_en_trop:
            st.warning(f" Colonnes ignorées (non utilisées par le modèle) : {list(colonnes_en_trop)}")
        
        # Sélectionner et réorganiser les colonnes dans le bon ordre
        df_for_prediction = df_new[colonnes_attendues]
        
        # Normalisation
        df_scaled = scaler.transform(df_for_prediction)
        
        # Prédiction
        predictions = model.predict(df_scaled)
        
        # Ajouter les prédictions au DataFrame original (avec toutes les colonnes)
        df_new["Prediction_Attack_type"] = predictions
        
        st.subheader("Résultats de la prédiction")
        st.dataframe(df_new.head())
        
        # Afficher la distribution des prédictions
        st.subheader("Distribution des types d'attaques détectées")
        prediction_counts = pd.Series(predictions).value_counts()
        st.bar_chart(prediction_counts)
        
        # Téléchargement
        csv = df_new.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger les résultats",
            csv,
            "predictions.csv",
            "text/csv",
            key="download-csv"
        )
        
        st.success(f" Prédiction réussie pour {len(df_new)} enregistrements")
        
    except AttributeError:
        # Si le scaler n'a pas feature_names_in_ (ancienne version sklearn)
        st.error(" Le scaler ne contient pas les noms de colonnes. Veuillez réentraîner le modèle avec une version récente de scikit-learn.")
        st.info("Tentative de prédiction avec les colonnes dans l'ordre actuel...")
        
        df_scaled = scaler.transform(df_new)
        predictions = model.predict(df_scaled)
        df_new["Prediction_Attack_type"] = predictions
        
        st.subheader("Résultats de la prédiction")
        st.dataframe(df_new.head())
        
        csv = df_new.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger les résultats",
            csv,
            "predictions.csv",
            "text/csv"
        )
        
    except Exception as e:
        st.error(f" Erreur lors de la prédiction : {str(e)}")
        st.write("Informations de débogage :")
        st.write(f"- Nombre de colonnes dans le fichier : {len(df_new.columns)}")
        st.write(f"- Colonnes du fichier : {df_new.columns.tolist()}")  
