===================
Guide d'utilisation
===================

Projet de détection des troubles du sommeil basé sur l'analyse de PSG
====================================================================

Ce guide détaillé explique comment utiliser le projet de détection des troubles du sommeil à partir de données polysomnographiques (PSG) au format `.edf`. Le système analyse les signaux physiologiques pour identifier trois troubles principaux : l'insomnie, l'apnée du sommeil et la narcolepsie.

Prérequis
=========

Avant de commencer, assurez-vous d'avoir :

- Un compte Kaggle avec accès aux notebooks
- Connaissance de base de Python et des notebooks Jupyter
- Un fichier PSG au format `.edf` (par exemple de la base de données Sleep-EDF)

Utilisation du notebook Kaggle
=============================

Le projet est entièrement implémenté dans un notebook Kaggle. Pour l'utiliser :

1. Accédez au notebook via le lien fourni ou copiez-le dans votre espace Kaggle
2. Téléchargez votre fichier `.edf` dans l'espace de données Kaggle
3. Exécutez les cellules du notebook dans l'ordre

Structure du notebook
===================

Le notebook est organisé en sections correspondant aux différentes étapes du pipeline de traitement :

::

    1. Configuration et importation des bibliothèques
    2. Chargement des fichiers .edf
    3. Transformation des signaux en CSV
    4. Prétraitement des données
    5. Prédiction des signaux (EEG et EMG)
    6. Détection des troubles du sommeil
    7. Visualisation des résultats

Étapes détaillées du pipeline
============================

1. Importation du fichier `.edf`
----------------------------------

Dans cette section du notebook, vous importez votre fichier PSG au format `.edf`.

*Code à exécuter dans le notebook* :

python

    import mne
    import os
    
    # Spécifiez le chemin vers votre fichier .edf
    edf_file = '../input/sleep-edf-database-expanded/SC4001E0-PSG.edf'
    
    # Chargement du fichier EDF
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    
    # Affichage des informations sur les canaux disponibles
    print(raw.info)
    
    # Liste des canaux disponibles
    print("Canaux disponibles:", raw.ch_names)

**Résultat attendu** :
- Affichage des canaux disponibles dans le fichier EDF
- Confirmation que les signaux EEG (Fpz-Cz), EMG et EOG sont présents
- Affichage des caractéristiques des signaux (fréquence d'échantillonnage, etc.)

**Conseils** :
- Vérifiez que les canaux EEG, EMG et EOG sont correctement identifiés
- Assurez-vous que la fréquence d'échantillonnage est suffisante (généralement 100 Hz)

2. Transformation en CSV
-----------------------

Cette section du notebook convertit les signaux bruts du fichier ``.edf`` en DataFrames pandas pour faciliter l'analyse.

**Code à exécuter** :

python

    import pandas as pd
    import numpy as np

    # Extraire les signaux en DataFrames
    eeg_data = raw.get_data(picks=['EEG Fpz-Cz'])
    emg_data = raw.get_data(picks=['EMG submental'])
    eog_data = raw.get_data(picks=['EOG horizontal'])

    # Créer des DataFrames pandas
    sampling_freq = raw.info['sfreq']
    times = np.arange(0, len(eeg_data[0])) / sampling_freq

    eeg_df = pd.DataFrame({
    'timestamp': times,
    'value': eeg_data[0],
    'channel': 'EEG Fpz-Cz'
    })

    emg_df = pd.DataFrame({
    'timestamp': times,
    'value': emg_data[0],
    'channel': 'EMG submental'
    })

    eog_df = pd.DataFrame({
    'timestamp': times,
    'value': eog_data[0],
    'channel': 'EOG horizontal'
    })

   # Sauvegarder en CSV si nécessaire (optionnel dans Kaggle)
   eeg_df.to_csv('eeg.csv', index=False)
   emg_df.to_csv('emg.csv', index=False)
   eog_df.to_csv('eog.csv', index=False)

   # Afficher les premières lignes pour vérification
   print("Données EEG:")
   print(eeg_df.head())

   # Fusionner tous les signaux dans une seule DataFrame
  dftotal = pd.concat([eeg_df, emg_df, eog_df], ignore_index=True)

  # Afficher les premières lignes de la DataFrame fusionnée
  print("\nDonnées fusionnées (dftotal):")
  print(dftotal.head())

  # Sauvegarder la DataFrame fusionnée en CSV si nécessaire
  dftotal.to_csv('all_signals.csv', index=False)

*Résultat attendu* :
- Création de trois DataFrames contenant les signaux EEG, EMG et EOG
- Structure des DataFrames avec colonnes : timestamp, value, channel
- Affichage des premières lignes de données pour vérification
- Creation d'une dataframe dftotal qui contient tous les signaux
3. Prétraitement des données
---------------------------

Cette section du notebook nettoie et prépare les données pour l'analyse.

*Code à exécuter* :

.. code-block:: python
    import statsmodels.api as sm
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    dftotal.isna().sum()
    dftotal.dropna(inplace=True)
    
    sm.qqplot(dftotal.Label, line='s')
    sm.qqplot(dftotal["EEG Fpz-Cz"], line='s')

    # Supposons que votre DataFrame est dftotal
    colonnes_signaux = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']

    # Initialiser le MinMaxScaler (par défaut, mise à l'échelle entre 0 et 1)
    scaler = MinMaxScaler()

    # Appliquer la normalisation aux colonnes sélectionnées
    dftotal[colonnes_signaux] = scaler.fit_transform(dftotal[colonnes_signaux])
    
    dftotal['EEG Fpz-Cz'].plot(title="EEG Fpz-Cz	",figsize=(20,5))
    plt.show()
    
    dftotal['EEG Pz-Oz'].plot(title="EEG Pz-Oz",figsize=(20,5))
    plt.show()
  
    dftotal['EOG horizontal'].plot(title="EOG horizontal",figsize=(20,5))
    plt.show() 


*Résultat attendu* :
- DataFrames prétraités avec :
  - Pas de valeurs manquantes
  - Signal normalisé 
  - Graphiques montrant les signaux  après prétraitement

4. Prédiction du signal EMG
--------------------------

Cette section présente une approche avancée pour analyser le signal EMG afin de détecter des mouvements anormaux. Le système utilise des techniques d'apprentissage profond pour modéliser les patterns complexes présents dans les signaux cérébraux.

### Composantes principales

1. *Prétraitement des données*: Normalisation des signaux EEG bruts
2. *Extraction de caractéristiques*: Calcul de métriques statistiques, fréquentielles et de complexité
3. *Modélisation hybride*: Combinaison de couches convolutives et LSTM bidirectionnelles
4. *Prédiction adaptative*: Génération de signaux EEG futurs avec maintien des propriétés statistiques

### Code d'implémentation

python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Input
from tensorflow.keras.layers import LayerNormalization, GaussianNoise, Concatenate, Add, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import seaborn as sns

# Configuration pour améliorer les performances
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Définir les paramètres de graines aléatoires pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# FIX: Vérifier si dftotal existe, sinon créer des données synthétiques pour test
try:
    eeg_data = dftotal['EMG submental'].values  # Utilisation de vos données
except NameError:
    print("Variable 'dftotal' not found. Creating synthetic data for testing...")
    # Générer des données EEG synthétiques pour test
    n_samples = 3000
    t = np.linspace(0, 30, n_samples)
    eeg_data = np.sin(0.5*t) + 0.2*np.sin(2.5*t) + 0.1*np.sin(7.5*t) + 0.05*np.random.randn(n_samples)

# Paramètres du modèle - optimisés pour préserver les fluctuations
sequence_length = 30  # Raccourci pour capturer les oscillations à plus haute fréquence
pred_length = 1  # Prédire un point à la fois
stride = 1  # Augmenter le nombre d'exemples d'entraînement
batch_size = 32  # Petit batch pour meilleur ajustement
epochs = 50  # Plus d'epochs pour un apprentissage plus fin

# Normalisation des données - StandardScaler peut être préférable pour préserver les variations
scaler = MinMaxScaler(feature_range=(0, 1))
eeg_data_scaled = scaler.fit_transform(eeg_data.reshape(-1, 1)).flatten()

# Échantillonnage des données
sample_size = min(150000, len(eeg_data_scaled))  # FIX: S'assurer que sample_size n'est pas supérieur à la taille des données
eeg_sample = eeg_data_scaled[:sample_size]

# --- FONCTION D'EXTRACTION DE CARACTÉRISTIQUES AMÉLIORÉE ---
def extract_features(data, seq_length):
    """Extraire des caractéristiques avancées du signal EEG"""
    features = []
    
    # FIX: Vérifier que les données ont une forme correcte
    if len(data.shape) > 1 and data.shape[0] == 1:
        data = data.flatten()
    
    # FIX: Vérifier que la séquence est assez longue
    if len(data) < seq_length:
        print(f"Warning: Data length {len(data)} is less than sequence length {seq_length}")
        # Padding si nécessaire
        if len(data) > 0:
            padding = np.zeros(seq_length - len(data))
            data = np.concatenate([data, padding])
        else:
            # Si data est vide, retourner un tableau vide avec la bonne forme
            return np.zeros((0, 11))
    
    for i in range(len(data) - seq_length + 1):
        segment = data[i:i+seq_length]
        
        # Caractéristiques statistiques
        mean = np.mean(segment)
        std = np.std(segment)
        min_val = np.min(segment)
        max_val = np.max(segment)
        range_val = max_val - min_val
        
        # Tendance et variation
        gradient = np.gradient(segment).mean()
        abs_gradient = np.abs(np.gradient(segment)).mean()  # Mesure de la variabilité globale
        
        # Analyse fréquentielle
        fft_vals = np.abs(np.fft.rfft(segment))
        # FIX: S'assurer que fft_vals a au moins un élément avant d'accéder à l'index 1
        if len(fft_vals) > 1:
            dominant_freq_idx = np.argmax(fft_vals[1:]) + 1  # Ignorer DC (indice 0)
            dominant_freq_val = fft_vals[dominant_freq_idx] / len(segment)
        else:
            dominant_freq_val = 0
        
        # Entropie approximative (mesure de la complexité/régularité)
        # Version simplifiée, considérez une implémentation plus complète au besoin
        diffs = np.diff(segment)
        direction_changes = np.sum(np.diff(np.signbit(diffs)) != 0)
        complexity = direction_changes / max(1, (seq_length - 2))  # FIX: Éviter division par zéro
        
        # Caractéristiques de forme d'onde
        peak_count = len(np.where(np.diff(np.signbit(np.diff(segment))) < 0)[0])  # Nombre de pics
        zero_crossings = len(np.where(np.diff(np.signbit(segment)))[0])  # Nombre de passages à zéro
        
        # Regrouper toutes les caractéristiques
        feature_vec = np.array([
            mean, std, min_val, max_val, range_val, 
            gradient, abs_gradient, 
            dominant_freq_val, complexity,
            peak_count / max(1, seq_length), zero_crossings / max(1, seq_length)  # FIX: Éviter division par zéro
        ])
        
        features.append(feature_vec)
    
    return np.array(features)

# --- CRÉATION DE SÉQUENCES AVEC CARACTÉRISTIQUES ENRICHIES ---
def create_enriched_sequences(data, seq_length, pred_length=1, stride=1):
    """Crée des séquences d'entrée/sortie avec des caractéristiques enrichies"""
    X_raw, y = [], []
    
    # FIX: Vérifier que les données sont suffisantes
    if len(data) <= seq_length + pred_length:
        print(f"Warning: Data length {len(data)} is not sufficient for sequence length {seq_length} and prediction length {pred_length}")
        return np.array([]), np.array([]), np.array([])
    
    for i in range(0, len(data) - seq_length - pred_length + 1, stride):
        X_raw.append(data[i:i + seq_length])
        if pred_length == 1:
            y.append(data[i + seq_length])
        else:
            y.append(data[i + seq_length:i + seq_length + pred_length])
    
    X_raw = np.array(X_raw)
    
    # FIX: Vérifier que X_raw n'est pas vide avant d'extraire les caractéristiques
    if len(X_raw) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Extraire des caractéristiques supplémentaires
    X_features = extract_features(data, seq_length)
    
    # FIX: S'assurer que X_features et X_raw ont le même nombre d'échantillons
    min_samples = min(len(X_raw), len(X_features))
    if min_samples == 0:
        return np.array([]), np.array([]), np.array([])
    
    X_raw = X_raw[:min_samples]
    X_features = X_features[:min_samples]
    y = np.array(y)[:min_samples]
    
    return X_raw, X_features, y

# Créer des séquences avec caractéristiques enrichies
X_raw, X_features, y = create_enriched_sequences(eeg_sample, sequence_length, pred_length, stride)

# FIX: Vérifier que les données ne sont pas vides
if len(X_raw) == 0 or len(X_features) == 0 or len(y) == 0:
    raise ValueError("Sequences could not be created. Check your data and parameters.")

X_raw = X_raw.reshape((X_raw.shape[0], X_raw.shape[1], 1))

print(f"Forme des données brutes: {X_raw.shape}")
print(f"Forme des caractéristiques: {X_features.shape}")
print(f"Forme des données de sortie: {y.shape}")

# Division en ensembles d'entraînement et de test
train_size = int(len(X_raw) * 0.8)
X_raw_train, X_raw_test = X_raw[:train_size], X_raw[train_size:]
X_features_train, X_features_test = X_features[:train_size], X_features[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# FIX: Vérifier qu'il y a suffisamment de données pour validation
if train_size < 5:
    raise ValueError(f"Not enough training data. Train size: {train_size}")

# Création d'un ensemble de validation
val_size = max(1, int(train_size * 0.2))  # FIX: Au moins 1 échantillon pour validation
X_raw_val, X_features_val, y_val = X_raw_train[-val_size:], X_features_train[-val_size:], y_train[-val_size:]
X_raw_train, X_features_train, y_train = X_raw_train[:-val_size], X_features_train[:-val_size], y_train[:-val_size]

# --- MODÈLE AVANCÉ POUR EEG ---
def create_advanced_emg_model(seq_length, feature_dim):
    """
    Modèle hybride optimisé pour capturer la complexité des signaux EMG
    """
    # Entrée pour la séquence brute
    input_seq = Input(shape=(seq_length, 1))
    
    # Entrée pour les caractéristiques extraites
    input_features = Input(shape=(feature_dim,))
    
    # --- Branche de convolution multi-échelle ---
    # Ajouter du bruit subtil pour améliorer la généralisation
    x = GaussianNoise(0.005)(input_seq)
    
    # Convolutions parallèles avec différentes tailles de noyau pour capturer différentes échelles
    conv_kernels = [2, 3, 5, 7]
    conv_outputs = []
    
    for kernel_size in conv_kernels:
        # FIX: S'assurer que kernel_size n'est pas plus grand que la séquence
        if kernel_size <= seq_length:
            conv = Conv1D(
                filters=32, 
                kernel_size=kernel_size,
                padding='same',
                activation='elu'  # ELU peut être meilleur pour préserver les variations subtiles
            )(x)
            conv = LayerNormalization()(conv)  # Normalisation pour stabiliser l'entraînement
            conv_outputs.append(conv)
    
    # FIX: Vérifier qu'il y a des sorties de convolution
    if not conv_outputs:
        raise ValueError(f"All kernel sizes {conv_kernels} are larger than sequence length {seq_length}")
        
    # Concaténer les sorties de convolution
    x = Concatenate()(conv_outputs)
    
    # Convolutions dilatées pour capturer des dépendances à long terme
    dilation_rates = [1, 2, 4, 8]
    dilated_outputs = []
    
    for dilation_rate in dilation_rates:
        # FIX: Vérifier que la dilatation n'est pas trop grande pour la séquence
        if (3 - 1) * dilation_rate + 1 <= seq_length:  # Taille effective = (kernel_size - 1) * dilation_rate + 1
            dilated_conv = Conv1D(
                filters=32, 
                kernel_size=3, 
                padding='causal', 
                dilation_rate=dilation_rate,
                activation='elu'
            )(x)
            dilated_outputs.append(dilated_conv)
    
    # FIX: Vérifier qu'il y a des sorties dilatées
    if not dilated_outputs:
        # Utiliser une convolution simple si aucune dilatation ne convient
        dilated_conv = Conv1D(
            filters=32, 
            kernel_size=1,  # Kernel size 1 fonctionnera toujours
            activation='elu'
        )(x)
        dilated_outputs.append(dilated_conv)
        
    # Combiner les convolutions dilatées
    x = Concatenate()(dilated_outputs)
    x = Conv1D(64, kernel_size=1, activation='elu')(x)  # Réduction de dimension
    
    # --- Branche LSTM bidirectionnelle ---
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm_out = LayerNormalization()(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)  # Dropout un peu plus élevé pour éviter le surapprentissage
    lstm_out = Bidirectional(LSTM(48, return_sequences=False))(lstm_out)
    
    # --- Traitement des caractéristiques ---
    # Passer les caractéristiques par des couches denses pour un meilleur apprentissage
    features_dense = Dense(32, activation='elu')(input_features)
    features_dense = Dense(32, activation='elu')(features_dense)
    
    # --- Fusion des branches ---
    combined = Concatenate()([lstm_out, features_dense])
    
    # Couches de sortie 
    x = Dense(64, activation='elu')(combined)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='elu')(x)
    
    # Couche de sortie - pas d'activation pour la régression
    output = Dense(pred_length)(x)
    
    # Créer le modèle
    model = Model(inputs=[input_seq, input_features], outputs=output)
    
    # Optimiseur avec taux d'apprentissage adaptatif et clippage de gradient
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

# Créer le modèle amélioré
model = create_advanced_emg_model(sequence_length, X_features.shape[1])
model.summary()

# Callbacks pour l'entraînement avec sauvegarde et réduction du taux d'apprentissage
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)
model_checkpoint = ModelCheckpoint(
    'best_emg_model.keras', 
    save_best_only=True, 
    monitor='val_loss',
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=6, 
    min_lr=0.00001,
    verbose=1
)

# Entraînement du modèle
history = model.fit(
    [X_raw_train, X_features_train], y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([X_raw_val, X_features_val], y_val),
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# Visualisation de la courbe d'apprentissage
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Évolution de la perte pendant l\'entraînement')
plt.xlabel('Époques')
plt.ylabel('Perte (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('emg_training_loss.png')
plt.close()

# FIX: Vérifier qu'il y a des données de test avant de prédire
if len(X_raw_test) == 0:
    raise ValueError("No test data available")

# Prédiction sur l'ensemble de test
y_pred = model.predict([X_raw_test, X_features_test])

# Inverser la normalisation pour obtenir les valeurs réelles
if pred_length == 1:
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
else:
    y_test_inv = np.array([scaler.inverse_transform(y.reshape(-1, 1)).flatten() for y in y_test])
    y_pred_inv = np.array([scaler.inverse_transform(y.reshape(-1, 1)).flatten() for y in y_pred])

# Calcul de l'erreur
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
print(f"Erreur quadratique moyenne (MSE): {mse}")
print(f"Racine de l'erreur quadratique moyenne (RMSE): {rmse}")

# --- AMÉLIORATION MAJEURE: PRÉVISION RÉALISTE AVEC RECALCUL DES CARACTÉRISTIQUES ---
def forecast_realistic_emg(model, initial_sequence, steps_ahead=200, scaler=None):
    """
    Prévision réaliste de signal EMG qui recalcule les caractéristiques à chaque étape
    et utilise une approche adaptative pour maintenir la naturalité du signal
    """
    sequence_length = initial_sequence.shape[0]
    future_predictions = []
    
    # Séquence courante pour la prédiction itérative
    current_sequence = initial_sequence.flatten().copy()
    
    # FIX: Vérifier qu'il y a assez de points dans la séquence
    if len(current_sequence) < 10:
        print(f"Warning: Initial sequence has only {len(current_sequence)} points, padding with zeros")
        padding = np.zeros(10 - len(current_sequence))
        current_sequence = np.concatenate([current_sequence, padding])
    
    # Historique récent des variations pour maintenir la cohérence
    recent_variations = np.diff(current_sequence[-10:])
    variation_history = np.abs(recent_variations).mean()
    
    for i in range(steps_ahead):
        # FIX: Vérifier que la séquence actuelle est de la bonne longueur
        if len(current_sequence) < sequence_length:
            padding = np.zeros(sequence_length - len(current_sequence))
            temp_sequence = np.concatenate([current_sequence, padding])
        else:
            temp_sequence = current_sequence[-sequence_length:]
        
        # Extraire les caractéristiques de la séquence actuelle
        try:
            # FIX: S'assurer que extract_features reçoit un tableau non vide
            if len(temp_sequence) == 0:
                raise ValueError("Empty sequence for feature extraction")
                
            current_features_array = extract_features(temp_sequence, sequence_length)
            
            # FIX: Vérifier que les caractéristiques sont extraites correctement
            if len(current_features_array) == 0:
                raise ValueError("Feature extraction returned empty array")
                
            current_features = current_features_array[0]
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Utiliser des caractéristiques par défaut
            current_features = np.zeros(X_features.shape[1])
        
        # Prédire le prochain point
        next_point_scaled = model.predict(
            [temp_sequence.reshape(1, sequence_length, 1), 
             current_features.reshape(1, -1)]
        )[0][0]
        
        # Stabilité adaptative: si la prédiction s'écarte trop de la tendance récente,
        # appliquer une correction subtile
        last_point = current_sequence[-1]
        max_jump = 2.5 * variation_history  # Limite basée sur la volatilité historique
        
        # Limiter les sauts trop grands tout en préservant la direction
        if abs(next_point_scaled - last_point) > max_jump:
            direction = np.sign(next_point_scaled - last_point)
            next_point_scaled = last_point + direction * max_jump
        
        # Ajouter un peu de bruit adaptatif basé sur la volatilité du signal
        # Plus le signal était volatile, plus on permet de fluctuations
        noise_scale = variation_history * 0.5  # La fluctuation est proportionnelle à la volatilité historique
        # FIX: Gérer le cas où variation_history est 0 ou NaN
        if not np.isfinite(noise_scale) or noise_scale == 0:
            noise_scale = 0.01  # Valeur par défaut
            
        noise = np.random.normal(0, noise_scale)
        
        # S'assurer que le bruit ne fait pas sortir la valeur de l'intervalle [0,1]
        next_point_with_noise = np.clip(next_point_scaled + noise, 0, 1)
        
        # Ajouter à nos prédictions
        future_predictions.append(next_point_with_noise)
        
        # Mettre à jour la séquence pour la prochaine prédiction
        current_sequence = np.append(current_sequence[1:], next_point_with_noise)
        
        # Mettre à jour l'historique des variations pour l'adaptativité
        if i >= 1:
            new_variation = abs(future_predictions[-1] - future_predictions[-2])
            # Moyenne mobile exponentielle pour l'historique des variations
            variation_history = 0.85 * variation_history + 0.15 * new_variation
    
    # Convertir les prédictions en array numpy
    future_predictions = np.array(future_predictions)
    
    # Inverser la normalisation si un scaler est fourni
    if scaler is not None:
        future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1)).flatten()
    
    return future_predictions

# FIX: Vérifier qu'il y a des données de test avant de visualiser
if len(y_test_inv) < 100:
    print("Warning: Not enough test data for visualization. Using available data.")
    visualize_length = min(5, len(y_test_inv))
else:
    visualize_length = 5

# Visualiser quelques exemples de séquences réelles pour comprendre la variabilité
plt.figure(figsize=(15, 5))
for i in range(visualize_length):  # Afficher 5 séquences d'exemple (ou moins si pas assez de données)
    # FIX: S'assurer que l'index est valide
    max_start = max(0, len(y_test_inv) - 100)
    if max_start > 0:
        start_idx = np.random.randint(0, max_start)
        plot_length = min(100, len(y_test_inv) - start_idx)
        plt.plot(y_test_inv[start_idx:start_idx+plot_length], alpha=0.7)
    else:
        # Si pas assez de données, utiliser tout ce qui est disponible
        plt.plot(y_test_inv, alpha=0.7)

plt.title('Exemples de séquences EMG réelles (pour comprendre la variabilité)')
plt.xlabel('Points temporels')
plt.ylabel('Amplitude EMG')
plt.grid(True)
plt.savefig('emg_real_patterns.png')
plt.close()

# FIX: Vérifier qu'il y a des données de test avant de prendre la dernière séquence
if len(X_raw_test) > 0:
    # Prendre la dernière séquence comme point de départ
    last_known_sequence = X_raw_test[-1]
else:
    # S'il n'y a pas de données de test, utiliser la dernière séquence d'entraînement
    last_known_sequence = X_raw_train[-1]

# Prédire les points futurs avec la nouvelle méthode réaliste
future_steps = 200
future_predictions = forecast_realistic_emg(model, last_known_sequence, future_steps, scaler)

# FIX: Vérifier qu'il y a assez de données réelles pour la visualisation
if len(y_test_inv) < 100:
    real_history_length = len(y_test_inv)
    print(f"Warning: Only {real_history_length} real data points available for visualization")
else:
    real_history_length = 100

# Visualiser les prédictions futures réalistes
plt.figure(figsize=(15, 6))

# Afficher les dernières valeurs réelles disponibles
real_history = y_test_inv[-real_history_length:]
plt.plot(range(real_history_length), real_history, label='Données EMG réelles', color='blue')

# Afficher les prédictions futures
plt.plot(range(real_history_length-1, real_history_length-1+future_steps), future_predictions, 
         label='Prédictions futures', color='red', linestyle='--')
plt.axvline(x=real_history_length-1, color='green', linestyle='-', label='Présent')

plt.title('Prévision future réaliste du signal EMG')
plt.xlabel('Points temporels')
plt.ylabel('Amplitude EMG')
plt.legend()
plt.grid(True)
plt.savefig('emg_future_realistic_forecast.png')

# Analyse supplémentaire: comparer la distribution statistique
plt.figure(figsize=(12, 5))

# Configurer la grille pour deux graphiques côte à côte
plt.subplot(1, 2, 1)
plt.hist(real_history, bins=20, alpha=0.7, label='Données réelles')
plt.hist(future_predictions, bins=20, alpha=0.7, label='Prédictions')
plt.title('Distribution des valeurs')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(np.diff(real_history), bins=20, alpha=0.7, label='Différences réelles')
plt.hist(np.diff(future_predictions), bins=20, alpha=0.7, label='Différences prédites')
plt.title('Distribution des variations point-à-point')
plt.legend()

plt.tight_layout()
plt.savefig('emg_distribution_comparison.png')

print("Modèle de prévision EMG amélioré terminé. Les graphiques ont été sauvegardés.")
# Sauvegarder les prédictions et les vraies valeurs dans un DataFrame
results_df = pd.DataFrame({
    'y_true': y_test_inv,
    'y_pred': y_pred_inv
})

# Sauvegarder dans un fichier CSV
results_df.to_csv('emg_predictions.csv', index=False)

====================================================================
Ou bien vous pouvez telecharger le model deja entrainer via le lien 👉 [Télécharger le modèle (.keras)](https://drive.google.com/file/d/1WQmU3ywLMbfQma6Q_eG_GXgkme735jtM/view?usp=sharing)
====================================================================
### Explication technique des composantes principales

#### 1. Extraction de caractéristiques avancées
La fonction extract_features() calcule plusieurs métriques importantes sur les segments de signal EEG:

- *Caractéristiques statistiques*: moyenne, écart-type, minimum, maximum et amplitude du signal
- *Analyse de tendance*: calcul des gradients moyens pour détecter les changements de direction
- *Analyse fréquentielle*: transformée de Fourier pour identifier les fréquences dominantes
- *Mesures de complexité*: calcul approximatif de l'entropie et des changements de direction
- *Forme d'onde*: détection des pics et des passages par zéro pour caractériser l'oscillation

#### 2. Architecture du modèle hybride
Le modèle neuronal combine plusieurs approches complémentaires:

- *Convolutions multi-échelles*: utilisation de noyaux de tailles variées (2, 3, 5, 7) pour capturer différentes échelles temporelles
- *Convolutions dilatées*: permet de percevoir des dépendances à long terme avec des taux de dilatation croissants (1, 2, 4, 8)
- *LSTM bidirectionnels*: analyse la séquence dans les deux directions temporelles pour mieux comprendre le contexte
- *Fusion multi-modalité*: combine les signaux bruts et les caractéristiques extraites pour une prédiction plus robuste

#### 3. Prédiction réaliste avec maintien de statistiques
La fonction forecast_realistic_emg() propose une méthode avancée pour générer des prédictions futures:

- *Prédiction adaptative*: recalcule les caractéristiques à chaque pas de temps pour plus de précision
- *Stabilité statistique*: limite les variations excessives en fonction de l'historique de volatilité
- *Bruit cohérent*: ajoute un bruit adaptatif pour maintenir la naturalité du signal
- *Moyenne mobile exponentielle*: met à jour dynamiquement l'estimation de la volatilité

### Résultats et visualisations
Le code génère plusieurs visualisations pour évaluer la qualité des prédictions:

1. Courbe d'apprentissage montrant l'évolution de la perte d'entraînement et de validation
2. Échantillons de séquences EEG réelles pour comprendre la variabilité naturelle
3. Prédiction future avec comparaison aux données historiques

5. Prédiction du signal EEG
--------------------------

Cette section présente une approche avancée pour analyser le signal EEG afin de détecter des mouvements anormaux. Le système utilise des techniques d'apprentissage profond pour modéliser les patterns complexes présents dans les signaux cérébraux.

### Composantes principales

1. *Prétraitement des données*: Normalisation des signaux EEG bruts
2. *Extraction de caractéristiques*: Calcul de métriques statistiques, fréquentielles et de complexité
3. *Modélisation hybride*: Combinaison de couches convolutives et LSTM bidirectionnelles
4. *Prédiction adaptative*: Génération de signaux EEG futurs avec maintien des propriétés statistiques

### Code d'implémentation

python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Input
from tensorflow.keras.layers import LayerNormalization, GaussianNoise, Concatenate, Add, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import seaborn as sns

# Configuration pour améliorer les performances
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Définir les paramètres de graines aléatoires pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# FIX: Vérifier si dftotal existe, sinon créer des données synthétiques pour test
try:
    eeg_data = dftotal['EEG Fpz-Cz'].values  # Utilisation de vos données
except NameError:
    print("Variable 'dftotal' not found. Creating synthetic data for testing...")
    # Générer des données EEG synthétiques pour test
    n_samples = 3000
    t = np.linspace(0, 30, n_samples)
    eeg_data = np.sin(0.5*t) + 0.2*np.sin(2.5*t) + 0.1*np.sin(7.5*t) + 0.05*np.random.randn(n_samples)

# Paramètres du modèle - optimisés pour préserver les fluctuations
sequence_length = 30  # Raccourci pour capturer les oscillations à plus haute fréquence
pred_length = 1  # Prédire un point à la fois
stride = 1  # Augmenter le nombre d'exemples d'entraînement
batch_size = 32  # Petit batch pour meilleur ajustement
epochs = 50  # Plus d'epochs pour un apprentissage plus fin

# Normalisation des données - StandardScaler peut être préférable pour préserver les variations
scaler = MinMaxScaler(feature_range=(0, 1))
eeg_data_scaled = scaler.fit_transform(eeg_data.reshape(-1, 1)).flatten()

# Échantillonnage des données
sample_size = min(150000, len(eeg_data_scaled))  # FIX: S'assurer que sample_size n'est pas supérieur à la taille des données
eeg_sample = eeg_data_scaled[:sample_size]

# --- FONCTION D'EXTRACTION DE CARACTÉRISTIQUES AMÉLIORÉE ---
def extract_features(data, seq_length):
    """Extraire des caractéristiques avancées du signal EEG"""
    features = []
    
    # FIX: Vérifier que les données ont une forme correcte
    if len(data.shape) > 1 and data.shape[0] == 1:
        data = data.flatten()
    
    # FIX: Vérifier que la séquence est assez longue
    if len(data) < seq_length:
        print(f"Warning: Data length {len(data)} is less than sequence length {seq_length}")
        # Padding si nécessaire
        if len(data) > 0:
            padding = np.zeros(seq_length - len(data))
            data = np.concatenate([data, padding])
        else:
            # Si data est vide, retourner un tableau vide avec la bonne forme
            return np.zeros((0, 11))
    
    for i in range(len(data) - seq_length + 1):
        segment = data[i:i+seq_length]
        
        # Caractéristiques statistiques
        mean = np.mean(segment)
        std = np.std(segment)
        min_val = np.min(segment)
        max_val = np.max(segment)
        range_val = max_val - min_val
        
        # Tendance et variation
        gradient = np.gradient(segment).mean()
        abs_gradient = np.abs(np.gradient(segment)).mean()  # Mesure de la variabilité globale
        
        # Analyse fréquentielle
        fft_vals = np.abs(np.fft.rfft(segment))
        # FIX: S'assurer que fft_vals a au moins un élément avant d'accéder à l'index 1
        if len(fft_vals) > 1:
            dominant_freq_idx = np.argmax(fft_vals[1:]) + 1  # Ignorer DC (indice 0)
            dominant_freq_val = fft_vals[dominant_freq_idx] / len(segment)
        else:
            dominant_freq_val = 0
        
        # Entropie approximative (mesure de la complexité/régularité)
        # Version simplifiée, considérez une implémentation plus complète au besoin
        diffs = np.diff(segment)
        direction_changes = np.sum(np.diff(np.signbit(diffs)) != 0)
        complexity = direction_changes / max(1, (seq_length - 2))  # FIX: Éviter division par zéro
        
        # Caractéristiques de forme d'onde
        peak_count = len(np.where(np.diff(np.signbit(np.diff(segment))) < 0)[0])  # Nombre de pics
        zero_crossings = len(np.where(np.diff(np.signbit(segment)))[0])  # Nombre de passages à zéro
        
        # Regrouper toutes les caractéristiques
        feature_vec = np.array([
            mean, std, min_val, max_val, range_val, 
            gradient, abs_gradient, 
            dominant_freq_val, complexity,
            peak_count / max(1, seq_length), zero_crossings / max(1, seq_length)  # FIX: Éviter division par zéro
        ])
        
        features.append(feature_vec)
    
    return np.array(features)

# --- CRÉATION DE SÉQUENCES AVEC CARACTÉRISTIQUES ENRICHIES ---
def create_enriched_sequences(data, seq_length, pred_length=1, stride=1):
    """Crée des séquences d'entrée/sortie avec des caractéristiques enrichies"""
    X_raw, y = [], []
    
    # FIX: Vérifier que les données sont suffisantes
    if len(data) <= seq_length + pred_length:
        print(f"Warning: Data length {len(data)} is not sufficient for sequence length {seq_length} and prediction length {pred_length}")
        return np.array([]), np.array([]), np.array([])
    
    for i in range(0, len(data) - seq_length - pred_length + 1, stride):
        X_raw.append(data[i:i + seq_length])
        if pred_length == 1:
            y.append(data[i + seq_length])
        else:
            y.append(data[i + seq_length:i + seq_length + pred_length])
    
    X_raw = np.array(X_raw)
    
    # FIX: Vérifier que X_raw n'est pas vide avant d'extraire les caractéristiques
    if len(X_raw) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Extraire des caractéristiques supplémentaires
    X_features = extract_features(data, seq_length)
    
    # FIX: S'assurer que X_features et X_raw ont le même nombre d'échantillons
    min_samples = min(len(X_raw), len(X_features))
    if min_samples == 0:
        return np.array([]), np.array([]), np.array([])
    
    X_raw = X_raw[:min_samples]
    X_features = X_features[:min_samples]
    y = np.array(y)[:min_samples]
    
    return X_raw, X_features, y

# Créer des séquences avec caractéristiques enrichies
X_raw, X_features, y = create_enriched_sequences(eeg_sample, sequence_length, pred_length, stride)

# FIX: Vérifier que les données ne sont pas vides
if len(X_raw) == 0 or len(X_features) == 0 or len(y) == 0:
    raise ValueError("Sequences could not be created. Check your data and parameters.")

X_raw = X_raw.reshape((X_raw.shape[0], X_raw.shape[1], 1))

print(f"Forme des données brutes: {X_raw.shape}")
print(f"Forme des caractéristiques: {X_features.shape}")
print(f"Forme des données de sortie: {y.shape}")

# Division en ensembles d'entraînement et de test
train_size = int(len(X_raw) * 0.8)
X_raw_train, X_raw_test = X_raw[:train_size], X_raw[train_size:]
X_features_train, X_features_test = X_features[:train_size], X_features[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# FIX: Vérifier qu'il y a suffisamment de données pour validation
if train_size < 5:
    raise ValueError(f"Not enough training data. Train size: {train_size}")

# Création d'un ensemble de validation
val_size = max(1, int(train_size * 0.2))  # FIX: Au moins 1 échantillon pour validation
X_raw_val, X_features_val, y_val = X_raw_train[-val_size:], X_features_train[-val_size:], y_train[-val_size:]
X_raw_train, X_features_train, y_train = X_raw_train[:-val_size], X_features_train[:-val_size], y_train[:-val_size]

# --- MODÈLE AVANCÉ POUR EEG ---
def create_advanced_eeg_model(seq_length, feature_dim):
    """
    Modèle hybride optimisé pour capturer la complexité des signaux EEG
    """
    # Entrée pour la séquence brute
    input_seq = Input(shape=(seq_length, 1))
    
    # Entrée pour les caractéristiques extraites
    input_features = Input(shape=(feature_dim,))
    
    # --- Branche de convolution multi-échelle ---
    # Ajouter du bruit subtil pour améliorer la généralisation
    x = GaussianNoise(0.005)(input_seq)
    
    # Convolutions parallèles avec différentes tailles de noyau pour capturer différentes échelles
    conv_kernels = [2, 3, 5, 7]
    conv_outputs = []
    
    for kernel_size in conv_kernels:
        # FIX: S'assurer que kernel_size n'est pas plus grand que la séquence
        if kernel_size <= seq_length:
            conv = Conv1D(
                filters=32, 
                kernel_size=kernel_size,
                padding='same',
                activation='elu'  # ELU peut être meilleur pour préserver les variations subtiles
            )(x)
            conv = LayerNormalization()(conv)  # Normalisation pour stabiliser l'entraînement
            conv_outputs.append(conv)
    
    # FIX: Vérifier qu'il y a des sorties de convolution
    if not conv_outputs:
        raise ValueError(f"All kernel sizes {conv_kernels} are larger than sequence length {seq_length}")
        
    # Concaténer les sorties de convolution
    x = Concatenate()(conv_outputs)
    
    # Convolutions dilatées pour capturer des dépendances à long terme
    dilation_rates = [1, 2, 4, 8]
    dilated_outputs = []
    
    for dilation_rate in dilation_rates:
        # FIX: Vérifier que la dilatation n'est pas trop grande pour la séquence
        if (3 - 1) * dilation_rate + 1 <= seq_length:  # Taille effective = (kernel_size - 1) * dilation_rate + 1
            dilated_conv = Conv1D(
                filters=32, 
                kernel_size=3, 
                padding='causal', 
                dilation_rate=dilation_rate,
                activation='elu'
            )(x)
            dilated_outputs.append(dilated_conv)
    
    # FIX: Vérifier qu'il y a des sorties dilatées
    if not dilated_outputs:
        # Utiliser une convolution simple si aucune dilatation ne convient
        dilated_conv = Conv1D(
            filters=32, 
            kernel_size=1,  # Kernel size 1 fonctionnera toujours
            activation='elu'
        )(x)
        dilated_outputs.append(dilated_conv)
        
    # Combiner les convolutions dilatées
    x = Concatenate()(dilated_outputs)
    x = Conv1D(64, kernel_size=1, activation='elu')(x)  # Réduction de dimension
    
    # --- Branche LSTM bidirectionnelle ---
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm_out = LayerNormalization()(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)  # Dropout un peu plus élevé pour éviter le surapprentissage
    lstm_out = Bidirectional(LSTM(48, return_sequences=False))(lstm_out)
    
    # --- Traitement des caractéristiques ---
    # Passer les caractéristiques par des couches denses pour un meilleur apprentissage
    features_dense = Dense(32, activation='elu')(input_features)
    features_dense = Dense(32, activation='elu')(features_dense)
    
    # --- Fusion des branches ---
    combined = Concatenate()([lstm_out, features_dense])
    
    # Couches de sortie 
    x = Dense(64, activation='elu')(combined)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='elu')(x)
    
    # Couche de sortie - pas d'activation pour la régression
    output = Dense(pred_length)(x)
    
    # Créer le modèle
    model = Model(inputs=[input_seq, input_features], outputs=output)
    
    # Optimiseur avec taux d'apprentissage adaptatif et clippage de gradient
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

# Créer le modèle amélioré
model = create_advanced_eeg_model(sequence_length, X_features.shape[1])
model.summary()

# Callbacks pour l'entraînement avec sauvegarde et réduction du taux d'apprentissage
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)
model_checkpoint = ModelCheckpoint(
    'best_eeg_model.keras', 
    save_best_only=True, 
    monitor='val_loss',
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=6, 
    min_lr=0.00001,
    verbose=1
)

# Entraînement du modèle
history = model.fit(
    [X_raw_train, X_features_train], y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([X_raw_val, X_features_val], y_val),
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# Visualisation de la courbe d'apprentissage
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Évolution de la perte pendant l\'entraînement')
plt.xlabel('Époques')
plt.ylabel('Perte (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('eeg_training_loss.png')
plt.close()

# FIX: Vérifier qu'il y a des données de test avant de prédire
if len(X_raw_test) == 0:
    raise ValueError("No test data available")

# Prédiction sur l'ensemble de test
y_pred = model.predict([X_raw_test, X_features_test])

# Inverser la normalisation pour obtenir les valeurs réelles
if pred_length == 1:
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
else:
    y_test_inv = np.array([scaler.inverse_transform(y.reshape(-1, 1)).flatten() for y in y_test])
    y_pred_inv = np.array([scaler.inverse_transform(y.reshape(-1, 1)).flatten() for y in y_pred])

# Calcul de l'erreur
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
print(f"Erreur quadratique moyenne (MSE): {mse}")
print(f"Racine de l'erreur quadratique moyenne (RMSE): {rmse}")

# --- AMÉLIORATION MAJEURE: PRÉVISION RÉALISTE AVEC RECALCUL DES CARACTÉRISTIQUES ---
def forecast_realistic_eeg(model, initial_sequence, steps_ahead=200, scaler=None):
    """
    Prévision réaliste de signal EEG qui recalcule les caractéristiques à chaque étape
    et utilise une approche adaptative pour maintenir la naturalité du signal
    """
    sequence_length = initial_sequence.shape[0]
    future_predictions = []
    
    # Séquence courante pour la prédiction itérative
    current_sequence = initial_sequence.flatten().copy()
    
    # FIX: Vérifier qu'il y a assez de points dans la séquence
    if len(current_sequence) < 10:
        print(f"Warning: Initial sequence has only {len(current_sequence)} points, padding with zeros")
        padding = np.zeros(10 - len(current_sequence))
        current_sequence = np.concatenate([current_sequence, padding])
    
    # Historique récent des variations pour maintenir la cohérence
    recent_variations = np.diff(current_sequence[-10:])
    variation_history = np.abs(recent_variations).mean()
    
    for i in range(steps_ahead):
        # FIX: Vérifier que la séquence actuelle est de la bonne longueur
        if len(current_sequence) < sequence_length:
            padding = np.zeros(sequence_length - len(current_sequence))
            temp_sequence = np.concatenate([current_sequence, padding])
        else:
            temp_sequence = current_sequence[-sequence_length:]
        
        # Extraire les caractéristiques de la séquence actuelle
        try:
            # FIX: S'assurer que extract_features reçoit un tableau non vide
            if len(temp_sequence) == 0:
                raise ValueError("Empty sequence for feature extraction")
                
            current_features_array = extract_features(temp_sequence, sequence_length)
            
            # FIX: Vérifier que les caractéristiques sont extraites correctement
            if len(current_features_array) == 0:
                raise ValueError("Feature extraction returned empty array")
                
            current_features = current_features_array[0]
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Utiliser des caractéristiques par défaut
            current_features = np.zeros(X_features.shape[1])
        
        # Prédire le prochain point
        next_point_scaled = model.predict(
            [temp_sequence.reshape(1, sequence_length, 1), 
             current_features.reshape(1, -1)]
        )[0][0]
        
        # Stabilité adaptative: si la prédiction s'écarte trop de la tendance récente,
        # appliquer une correction subtile
        last_point = current_sequence[-1]
        max_jump = 2.5 * variation_history  # Limite basée sur la volatilité historique
        
        # Limiter les sauts trop grands tout en préservant la direction
        if abs(next_point_scaled - last_point) > max_jump:
            direction = np.sign(next_point_scaled - last_point)
            next_point_scaled = last_point + direction * max_jump
        
        # Ajouter un peu de bruit adaptatif basé sur la volatilité du signal
        # Plus le signal était volatile, plus on permet de fluctuations
        noise_scale = variation_history * 0.5  # La fluctuation est proportionnelle à la volatilité historique
        # FIX: Gérer le cas où variation_history est 0 ou NaN
        if not np.isfinite(noise_scale) or noise_scale == 0:
            noise_scale = 0.01  # Valeur par défaut
            
        noise = np.random.normal(0, noise_scale)
        
        # S'assurer que le bruit ne fait pas sortir la valeur de l'intervalle [0,1]
        next_point_with_noise = np.clip(next_point_scaled + noise, 0, 1)
        
        # Ajouter à nos prédictions
        future_predictions.append(next_point_with_noise)
        
        # Mettre à jour la séquence pour la prochaine prédiction
        current_sequence = np.append(current_sequence[1:], next_point_with_noise)
        
        # Mettre à jour l'historique des variations pour l'adaptativité
        if i >= 1:
            new_variation = abs(future_predictions[-1] - future_predictions[-2])
            # Moyenne mobile exponentielle pour l'historique des variations
            variation_history = 0.85 * variation_history + 0.15 * new_variation
    
    # Convertir les prédictions en array numpy
    future_predictions = np.array(future_predictions)
    
    # Inverser la normalisation si un scaler est fourni
    if scaler is not None:
        future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1)).flatten()
    
    return future_predictions

# FIX: Vérifier qu'il y a des données de test avant de visualiser
if len(y_test_inv) < 100:
    print("Warning: Not enough test data for visualization. Using available data.")
    visualize_length = min(5, len(y_test_inv))
else:
    visualize_length = 5

# Visualiser quelques exemples de séquences réelles pour comprendre la variabilité
plt.figure(figsize=(15, 5))
for i in range(visualize_length):  # Afficher 5 séquences d'exemple (ou moins si pas assez de données)
    # FIX: S'assurer que l'index est valide
    max_start = max(0, len(y_test_inv) - 100)
    if max_start > 0:
        start_idx = np.random.randint(0, max_start)
        plot_length = min(100, len(y_test_inv) - start_idx)
        plt.plot(y_test_inv[start_idx:start_idx+plot_length], alpha=0.7)
    else:
        # Si pas assez de données, utiliser tout ce qui est disponible
        plt.plot(y_test_inv, alpha=0.7)

plt.title('Exemples de séquences EEG réelles (pour comprendre la variabilité)')
plt.xlabel('Points temporels')
plt.ylabel('Amplitude EEG')
plt.grid(True)
plt.savefig('eeg_real_patterns.png')
plt.close()

# FIX: Vérifier qu'il y a des données de test avant de prendre la dernière séquence
if len(X_raw_test) > 0:
    # Prendre la dernière séquence comme point de départ
    last_known_sequence = X_raw_test[-1]
else:
    # S'il n'y a pas de données de test, utiliser la dernière séquence d'entraînement
    last_known_sequence = X_raw_train[-1]

# Prédire les points futurs avec la nouvelle méthode réaliste
future_steps = 200
future_predictions = forecast_realistic_eeg(model, last_known_sequence, future_steps, scaler)

# FIX: Vérifier qu'il y a assez de données réelles pour la visualisation
if len(y_test_inv) < 100:
    real_history_length = len(y_test_inv)
    print(f"Warning: Only {real_history_length} real data points available for visualization")
else:
    real_history_length = 100

# Visualiser les prédictions futures réalistes
plt.figure(figsize=(15, 6))

# Afficher les dernières valeurs réelles disponibles
real_history = y_test_inv[-real_history_length:]
plt.plot(range(real_history_length), real_history, label='Données EEG réelles', color='blue')

# Afficher les prédictions futures
plt.plot(range(real_history_length-1, real_history_length-1+future_steps), future_predictions, 
         label='Prédictions futures', color='red', linestyle='--')
plt.axvline(x=real_history_length-1, color='green', linestyle='-', label='Présent')

plt.title('Prévision future réaliste du signal EEG')
plt.xlabel('Points temporels')
plt.ylabel('Amplitude EEG')
plt.legend()
plt.grid(True)
plt.savefig('eeg_future_realistic_forecast.png')

# Analyse supplémentaire: comparer la distribution statistique
plt.figure(figsize=(12, 5))

# Configurer la grille pour deux graphiques côte à côte
plt.subplot(1, 2, 1)
plt.hist(real_history, bins=20, alpha=0.7, label='Données réelles')
plt.hist(future_predictions, bins=20, alpha=0.7, label='Prédictions')
plt.title('Distribution des valeurs')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(np.diff(real_history), bins=20, alpha=0.7, label='Différences réelles')
plt.hist(np.diff(future_predictions), bins=20, alpha=0.7, label='Différences prédites')
plt.title('Distribution des variations point-à-point')
plt.legend()

plt.tight_layout()
plt.savefig('eeg_distribution_comparison.png')
# Sauvegarder les prédictions et les vraies valeurs dans un DataFrame
results_df = pd.DataFrame({
    'y_true': y_test_inv,
    'y_pred': y_pred_inv
})

# Sauvegarder dans un fichier CSV
results_df.to_csv('eeg_predictions.csv', index=False)

print("Les prédictions ont été sauvegardées dans 'eeg_predictions.csv'")

print("Modèle de prévision EEG amélioré terminé. Les graphiques ont été sauvegardés.")

====================================================================
Ou bien vous pouvez telecharger le model deja entrainer via le lien 👉 [Télécharger le modèle (.keras)](https://drive.google.com/file/d/1ZKH9WgfoknVqFe2q4FvL_zb96o9EVRid/view?usp=sharing)
====================================================================
### Explication technique des composantes principales

#### 1. Extraction de caractéristiques avancées
La fonction extract_features() calcule plusieurs métriques importantes sur les segments de signal EEG:

- *Caractéristiques statistiques*: moyenne, écart-type, minimum, maximum et amplitude du signal
- *Analyse de tendance*: calcul des gradients moyens pour détecter les changements de direction
- *Analyse fréquentielle*: transformée de Fourier pour identifier les fréquences dominantes
- *Mesures de complexité*: calcul approximatif de l'entropie et des changements de direction
- *Forme d'onde*: détection des pics et des passages par zéro pour caractériser l'oscillation

#### 2. Architecture du modèle hybride
Le modèle neuronal combine plusieurs approches complémentaires:

- *Convolutions multi-échelles*: utilisation de noyaux de tailles variées (2, 3, 5, 7) pour capturer différentes échelles temporelles
- *Convolutions dilatées*: permet de percevoir des dépendances à long terme avec des taux de dilatation croissants (1, 2, 4, 8)
- *LSTM bidirectionnels*: analyse la séquence dans les deux directions temporelles pour mieux comprendre le contexte
- *Fusion multi-modalité*: combine les signaux bruts et les caractéristiques extraites pour une prédiction plus robuste

#### 3. Prédiction réaliste avec maintien de statistiques
La fonction forecast_realistic_eeg() propose une méthode avancée pour générer des prédictions futures:

- *Prédiction adaptative*: recalcule les caractéristiques à chaque pas de temps pour plus de précision
- *Stabilité statistique*: limite les variations excessives en fonction de l'historique de volatilité
- *Bruit cohérent*: ajoute un bruit adaptatif pour maintenir la naturalité du signal
- *Moyenne mobile exponentielle*: met à jour dynamiquement l'estimation de la volatilité

### Résultats et visualisations
Le code génère plusieurs visualisations pour évaluer la qualité des prédictions:

1. Courbe d'apprentissage montrant l'évolution de la perte d'entraînement et de validation
2. Échantillons de séquences EEG réelles pour comprendre la variabilité naturelle
3. Prédiction future avec comparaison aux données historiques

6. Détection des troubles du sommeil
-----------------------------------

Cette section finale combine les analyses des signaux EEG et EMG pour identifier les troubles du sommeil.

*Code à exécuter* :


*Résultat attendu* :
- Graphique à barres montrant les probabilités de chaque trouble
- Visualisation des segments de signaux EEG et EMG
- Hypnogramme approximatif des stades de sommeil

Conseils d'optimisation
======================

Pour obtenir les meilleurs résultats :

1. *Qualité des données* :

   - Utilisez des enregistrements PSG complets (8h minimum)
   - Vérifiez que les signaux EEG et EMG sont clairement identifiés
   - Préférez les fichiers `.edf` standardisés (ex : Sleep-EDF database)

2. *Ajustement des paramètres du modèle* :

   - Augmentez le nombre d'époques pour un meilleur apprentissage
   - Testez différentes architectures LSTM (couches, unités)
   - Variez la taille des séquences d'entrée

3. *Adaptation des seuils de détection* :

   - Ajustez les seuils selon la sensibilité souhaitée
   - Pour un dépistage large : réduisez les seuils
   - Pour minimiser les faux positifs : augmentez les seuils

Dépannage
========

Problèmes courants et solutions
------------------------------

1. **Erreur lors du chargement du fichier `.edf`** :

   - *Problème*: MNE ne peut pas lire certains canaux
   - *Solution*: Vérifiez les noms des canaux avec `raw.ch_names` et ajustez le code

2. *Erreurs de prédiction* :

   - *Problème*: Modèle peu performant (MSE élevé)
   - *Solution*: Augmentez les données d'entraînement ou essayez d'autres architectures de modèle

3. *Mémoire insuffisante* :

   - *Problème*: Erreur de mémoire lors du traitement des grands fichiers
   - *Solution*: Réduisez la taille des batchs ou sous-échantillonnez les données

4. *Format de données incompatible* :

   - *Problème*: Structure de fichier EDF non standard
   - *Solution*: Utilisez des outils de conversion comme EDFbrowser avant l'importation

Ressources supplémentaires
=========================

- Sleep-EDF Database: https://physionet.org/content/sleep-edfx/1.0.0/
- Documentation MNE-Python: https://mne.tools/stable/index.html
- Tutoriel d'analyse du sommeil avec Python: https://raphaelvallat.com/yasa/build/html/index.html

Support
======

Si vous rencontrez des problèmes avec le notebook :

- Consultez les forums Kaggle pour des solutions similaires