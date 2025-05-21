# 💤 Sleep Disorder Detection using Time Series

Détection des troubles du sommeil (apnée, insomnie, narcolepsie) à partir de signaux physiologiques multicanaux (EEG, EMG, EOG) extraits de fichiers de polysomnographie (PSG).  
Ce projet applique des modèles de deep learning, notamment LSTM et attention, sur des séries temporelles pour identifier des troubles du sommeil de manière automatisée.

---

## 📌 Objectifs

- Extraire et prétraiter des signaux EEG/EMG/EOG depuis des fichiers `.edf`
- Prédire les stades de sommeil et détecter des troubles comme :
  - Apnée du sommeil
  - Insomnie
  - Narcolepsie
- Visualiser les signaux et les prédictions de manière interactive
- Proposer une interface utilisateur (Streamlit) simple d'utilisation

---

## 📁 Données utilisées

- **Dataset** : Sleep-EDF Expanded [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)
- **Signaux exploités** : EEG Fpz-Cz, EMG submental, EOG horizontal
- **Annotations** : stades de sommeil et événements d’apnée

---

## 🧠 Modèles de Deep Learning

- `LSTM` / `BiLSTM` pour la modélisation temporelle
- `Attention` pour la pondération contextuelle dans les séquences
- Classification multi-classes et binaire selon le trouble ciblé

---

## ⚙️ Technologies

- Python 3.10+
- NumPy, Pandas, SciPy
- MNE-Python (prétraitement EEG)
- TensorFlow / PyTorch
- Matplotlib, Seaborn
- Streamlit (interface web)

---

## 📦 Installation

```bash
git clone https://github.com/ton-utilisateur/sleep-disorder-detection.git
cd sleep-disorder-detection
pip install -r requirements.txt
