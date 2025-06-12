=====================================
Guide Complet d'Analyse de Journal de Sommeil
=====================================

Ce guide explique en détail chaque étape du code d'analyse de journal de sommeil, permettant de reproduire exactement les mêmes résultats.


Installation et Imports
=======================

Dépendances Requises
--------------------

Avant d'exécuter le code, installez toutes les bibliothèques nécessaires :

.. code-block:: bash

   pip install pandas numpy nltk scikit-learn matplotlib seaborn

Imports et Configuration
------------------------

Le code commence par importer toutes les bibliothèques nécessaires :

.. code-block:: python

   import pandas as pd
   import numpy as np
   import re
   import nltk
   from nltk.sentiment import SentimentIntensityAnalyzer
   from nltk.tokenize import word_tokenize, sent_tokenize
   from nltk.corpus import stopwords
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.cluster import KMeans
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report, accuracy_score
   import matplotlib.pyplot as plt
   import seaborn as sns
   from datetime import datetime, timedelta
   import warnings
   warnings.filterwarnings('ignore')

**Explication des imports :**
- **pandas/numpy** : Manipulation de données et calculs numériques
- **re** : Expressions régulières pour l'extraction d'entités
- **nltk** : Traitement du langage naturel et analyse de sentiment
- **sklearn** : Machine learning (vectorisation, clustering, classification)
- **matplotlib/seaborn** : Visualisation des données
- **datetime** : Gestion des dates et heures
- **warnings** : Suppression des avertissements pour une sortie plus propre

Téléchargement des Ressources NLTK
-----------------------------------

.. code-block:: python

   try:
       nltk.download('punkt', quiet=True)
       nltk.download('stopwords', quiet=True)
       nltk.download('vader_lexicon', quiet=True)
   except:
       pass

**Ressources téléchargées :**
- **punkt** : Tokenisation des phrases
- **stopwords** : Mots vides en français et anglais
- **vader_lexicon** : Lexique pour l'analyse de sentiment VADER

Classe SleepDiaryAnalyzer
=========================

Initialisation de la Classe
----------------------------

.. code-block:: python

   class SleepDiaryAnalyzer:
       def __init__(self):
           self.sia = SentimentIntensityAnalyzer()
           self.stop_words = set(stopwords.words('french') + stopwords.words('english'))
           self.vectorizer = TfidfVectorizer(max_features=100, stop_words='french')
           self.classifier = None

**Composants initialisés :**
- **sia** : Analyseur de sentiment VADER
- **stop_words** : Mots vides français et anglais combinés
- **vectorizer** : Vectoriseur TF-IDF limité à 100 caractéristiques
- **classifier** : Classificateur (initialisé à None)

Dictionnaires de Mots-Clés
---------------------------

.. code-block:: python

   self.sleep_keywords = {
       'fatigue': ['fatigué', 'épuisé', 'crevé', 'tired', 'exhausted', 'fatigué', 'las'],
       'stress': ['stressé', 'anxieux', 'angoissé', 'stressed', 'worried', 'inquiet'],
       'qualité_positive': ['reposé', 'frais', 'énergique', 'refreshed', 'energetic', 'bien dormi'],
       'qualité_negative': ['mal dormi', 'insomnie', 'réveils', 'cauchemar', 'agité'],
       'durée': ['heures', 'h', 'minutes', 'min', 'temps'],
       'réveils': ['réveillé', 'réveil', 'levé', 'debout', 'wake up', 'woke']
   }

**Catégories de mots-clés :**
- **fatigue** : Indicateurs de fatigue (français/anglais)
- **stress** : Indicateurs de stress et anxiété
- **qualité_positive** : Mots indiquant une bonne qualité de sommeil
- **qualité_negative** : Mots indiquant une mauvaise qualité
- **durée** : Mots liés à la durée du sommeil
- **réveils** : Mots liés aux réveils

Patterns d'Expression Régulière
--------------------------------

.. code-block:: python

   self.time_patterns = {
       'heures': r'(\d+)\s*(?:heures?|h)',
       'minutes': r'(\d+)\s*(?:minutes?|min)',
       'reveils': r'(\d+)\s*(?:fois|réveils?|wake)',
       'heure_coucher': r'(?:couché|dormi|sleep)\s*(?:à|at)\s*(\d{1,2})[h:]?(\d{0,2})',
       'heure_lever': r'(?:levé|réveillé|wake)\s*(?:à|at)\s*(\d{1,2})[h:]?(\d{0,2})'
   }

**Patterns définis :**
- **heures** : Capture "8 heures", "7h", etc.
- **minutes** : Capture "30 minutes", "45 min", etc.
- **reveils** : Capture "3 fois", "2 réveils", etc.
- **heure_coucher** : Capture l'heure de coucher
- **heure_lever** : Capture l'heure de lever

Méthodes de Préprocessing
=========================

Préprocessing du Texte
----------------------

.. code-block:: python

   def preprocess_text(self, text):
       """Préprocessing du texte"""
       if pd.isna(text):
           return ""
       
       text = text.lower()
       text = re.sub(r'[^\w\s]', ' ', text)
       text = re.sub(r'\s+', ' ', text).strip()
       return text

**Étapes du préprocessing :**
1. Vérification des valeurs nulles
2. Conversion en minuscules
3. Suppression de la ponctuation (garde uniquement mots et espaces)
4. Normalisation des espaces multiples
5. Suppression des espaces en début/fin

Extraction d'Entités Temporelles
---------------------------------

.. code-block:: python

   def extract_temporal_entities(self, text):
       """Extraction d'entités temporelles"""
       entities = {}
       text_lower = text.lower()
       
       # Extraction des heures de sommeil
       heures_match = re.search(self.time_patterns['heures'], text_lower)
       if heures_match:
           entities['heures_sommeil'] = int(heures_match.group(1))
       
       # Extraction des minutes
       minutes_match = re.search(self.time_patterns['minutes'], text_lower)
       if minutes_match:
           entities['minutes_sommeil'] = int(minutes_match.group(1))
       
       # Extraction du nombre de réveils
       reveils_match = re.search(self.time_patterns['reveils'], text_lower)
       if reveils_match:
           entities['nb_reveils'] = int(reveils_match.group(1))
       
       return entities

**Entités extraites :**
- **heures_sommeil** : Nombre d'heures de sommeil mentionnées
- **minutes_sommeil** : Minutes de sommeil supplémentaires
- **nb_reveils** : Nombre de réveils nocturnes

Analyse de Sentiment
====================

Méthode d'Analyse de Sentiment
-------------------------------

.. code-block:: python

   def analyze_sentiment(self, text):
       """Analyse de sentiment"""
       if not text:
           return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
       
       scores = self.sia.polarity_scores(text)
       return scores

**Scores retournés par VADER :**
- **compound** : Score global (-1 à +1)
- **pos** : Proportion de sentiment positif (0 à 1)
- **neu** : Proportion de sentiment neutre (0 à 1)
- **neg** : Proportion de sentiment négatif (0 à 1)

Extraction de Mots-Clés
========================

Comptage des Mots-Clés de Sommeil
----------------------------------

.. code-block:: python

   def extract_sleep_keywords(self, text):
       """Extraction des mots-clés liés au sommeil"""
       text_lower = text.lower()
       keyword_counts = {}
       
       for category, keywords in self.sleep_keywords.items():
           count = sum(1 for keyword in keywords if keyword in text_lower)
           keyword_counts[f'{category}_keywords'] = count
       
       return keyword_counts

**Processus :**
1. Conversion du texte en minuscules
2. Pour chaque catégorie de mots-clés :
   - Compter les occurrences de chaque mot-clé
   - Sommer les comptes par catégorie
3. Retourner un dictionnaire avec les comptes par catégorie

Classification de la Qualité du Sommeil
========================================

Algorithme de Classification
-----------------------------

.. code-block:: python

   def classify_sleep_quality(self, sentiment_scores, keywords, temporal_entities):
       """Classification de la qualité du sommeil"""
       score = 0
       
       # Score basé sur le sentiment
       score += sentiment_scores['compound'] * 2
       
       # Score basé sur les mots-clés
       score += keywords.get('qualité_positive_keywords', 0) * 0.5
       score -= keywords.get('qualité_negative_keywords', 0) * 0.5
       score -= keywords.get('fatigue_keywords', 0) * 0.3
       score -= keywords.get('stress_keywords', 0) * 0.4
       
       # Score basé sur les entités temporelles
       if 'heures_sommeil' in temporal_entities:
           heures = temporal_entities['heures_sommeil']
           if 7 <= heures <= 9:
               score += 0.5
           elif heures < 6 or heures > 10:
               score -= 0.5
       
       if 'nb_reveils' in temporal_entities:
           score -= temporal_entities['nb_reveils'] * 0.2
       
       # Classification
       if score > 0.5:
           return "Bonne qualité"
       elif score > -0.5:
           return "Qualité moyenne"
       else:
           return "Mauvaise qualité"

**Système de scoring :**
1. **Sentiment** : Score compound × 2 (poids élevé)
2. **Mots-clés positifs** : +0.5 par occurrence
3. **Mots-clés négatifs** : -0.5 par occurrence
4. **Fatigue** : -0.3 par occurrence
5. **Stress** : -0.4 par occurrence
6. **Heures optimales (7-9h)** : +0.5
7. **Heures sous-optimales (<6h ou >10h)** : -0.5
8. **Réveils** : -0.2 par réveil

**Seuils de classification :**
- Score > 0.5 : "Bonne qualité"
- Score > -0.5 : "Qualité moyenne"
- Score ≤ -0.5 : "Mauvaise qualité"

Détection de Problèmes de Sommeil
==================================

Identification des Troubles
----------------------------

.. code-block:: python

   def detect_sleep_issues(self, text, sentiment_scores, keywords, temporal_entities):
       """Détection de problèmes de sommeil spécifiques"""
       issues = []
       text_lower = text.lower()
       
       # Détection d'insomnie
       insomnia_indicators = ['insomnie', 'pas dormi', 'impossible de dormir', 'réveillé toute la nuit']
       if any(indicator in text_lower for indicator in insomnia_indicators):
           issues.append("Suspicion d'insomnie")
       
       # Détection de stress/anxiété
       if keywords.get('stress_keywords', 0) > 0 or sentiment_scores['neg'] > 0.6:
           issues.append("Stress/Anxiété détecté")
       
       # Détection de sommeil fragmenté
       if temporal_entities.get('nb_reveils', 0) >= 3:
           issues.append("Sommeil fragmenté")
       
       # Détection de durée insuffisante
       if temporal_entities.get('heures_sommeil', 8) < 6:
           issues.append("Durée de sommeil insuffisante")
       
       return issues

**Problèmes détectés :**
1. **Insomnie** : Mots-clés spécifiques d'insomnie
2. **Stress/Anxiété** : Mots-clés de stress OU sentiment négatif > 0.6
3. **Sommeil fragmenté** : ≥3 réveils nocturnes
4. **Durée insuffisante** : <6 heures de sommeil

Analyse Complète d'une Entrée
==============================

Méthode d'Analyse Intégrée
---------------------------

.. code-block:: python

   def analyze_single_entry(self, text):
       """Analyse complète d'une entrée de journal"""
       if not text or pd.isna(text):
           return None
       
       # Préprocessing
       processed_text = self.preprocess_text(text)
       
       # Analyses
       sentiment = self.analyze_sentiment(text)
       keywords = self.extract_sleep_keywords(text)
       temporal_entities = self.extract_temporal_entities(text)
       sleep_quality = self.classify_sleep_quality(sentiment, keywords, temporal_entities)
       sleep_issues = self.detect_sleep_issues(text, sentiment, keywords, temporal_entities)
       
       return {
           'texte_original': text,
           'texte_preprocessed': processed_text,
           'sentiment_compound': sentiment['compound'],
           'sentiment_positive': sentiment['pos'],
           'sentiment_negative': sentiment['neg'],
           'sentiment_neutral': sentiment['neu'],
           **keywords,
           **temporal_entities,
           'qualite_sommeil': sleep_quality,
           'problemes_detectes': sleep_issues,
           'score_global': sentiment['compound'] - keywords.get('fatigue_keywords', 0) * 0.1
       }

**Processus d'analyse :**
1. Vérification de la validité du texte
2. Préprocessing du texte
3. Analyse de sentiment
4. Extraction des mots-clés
5. Extraction des entités temporelles
6. Classification de la qualité
7. Détection des problèmes
8. Calcul du score global

**Données retournées :**
- Textes original et préprocessé
- Scores de sentiment (4 métriques)
- Comptes de mots-clés par catégorie
- Entités temporelles extraites
- Qualité du sommeil classifiée
- Liste des problèmes détectés
- Score global calculé

Analyse d'un Journal Complet
============================

Traitement de Multiple Entrées
-------------------------------

.. code-block:: python

   def analyze_sleep_diary(self, entries):
       """Analyse d'un journal de sommeil complet"""
       results = []
       
       for entry in entries:
           analysis = self.analyze_single_entry(entry)
           if analysis:
               results.append(analysis)
       
       return pd.DataFrame(results)

**Processus :**
1. Itération sur toutes les entrées
2. Analyse individuelle de chaque entrée
3. Collection des résultats non-nuls
4. Conversion en DataFrame pandas

Génération d'Insights
=====================

Calcul des Métriques Globales
------------------------------

.. code-block:: python

   def generate_insights(self, df_analysis):
       """Génération d'insights à partir des analyses"""
       if df_analysis.empty:
           return {}
       
       insights = {
           'total_entries': len(df_analysis),
           'avg_sentiment': df_analysis['sentiment_compound'].mean(),
           'quality_distribution': df_analysis['qualite_sommeil'].value_counts().to_dict(),
           'most_common_issues': [],
           'avg_sleep_hours': None,
           'avg_wake_ups': None
       }
       
       # Problèmes les plus fréquents
       all_issues = []
       for issues_list in df_analysis['problemes_detectes']:
           all_issues.extend(issues_list)
       
       if all_issues:
           from collections import Counter
           insights['most_common_issues'] = dict(Counter(all_issues).most_common(3))
       
       # Moyennes des entités temporelles
       if 'heures_sommeil' in df_analysis.columns:
           insights['avg_sleep_hours'] = df_analysis['heures_sommeil'].mean()
       
       if 'nb_reveils' in df_analysis.columns:
           insights['avg_wake_ups'] = df_analysis['nb_reveils'].mean()
       
       return insights

**Insights calculés :**
1. **total_entries** : Nombre total d'entrées
2. **avg_sentiment** : Sentiment moyen
3. **quality_distribution** : Distribution des qualités de sommeil
4. **most_common_issues** : Top 3 des problèmes les plus fréquents
5. **avg_sleep_hours** : Durée moyenne de sommeil
6. **avg_wake_ups** : Nombre moyen de réveils

Visualisations
==============

Création des Graphiques
------------------------

.. code-block:: python

   def create_visualizations(self, df_analysis):
       """Création de visualisations"""
       if df_analysis.empty:
           return None
       
       fig, axes = plt.subplots(2, 2, figsize=(15, 12))
       fig.suptitle('Analyse du Journal de Sommeil', fontsize=16, fontweight='bold')
       
       # 1. Distribution de la qualité du sommeil
       quality_counts = df_analysis['qualite_sommeil'].value_counts()
       axes[0, 0].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
       axes[0, 0].set_title('Distribution de la Qualité du Sommeil')
       
       # 2. Évolution du sentiment dans le temps
       axes[0, 1].plot(df_analysis.index, df_analysis['sentiment_compound'], marker='o')
       axes[0, 1].set_title('Évolution du Sentiment')
       axes[0, 1].set_xlabel('Entrées')
       axes[0, 1].set_ylabel('Score de Sentiment')
       axes[0, 1].grid(True, alpha=0.3)
       
       # 3. Corrélation sentiment vs mots-clés de fatigue
       if 'fatigue_keywords' in df_analysis.columns:
           axes[1, 0].scatter(df_analysis['fatigue_keywords'], df_analysis['sentiment_compound'])
           axes[1, 0].set_xlabel('Mots-clés de Fatigue')
           axes[1, 0].set_ylabel('Score de Sentiment')
           axes[1, 0].set_title('Fatigue vs Sentiment')
       
       # 4. Distribution des heures de sommeil
       if 'heures_sommeil' in df_analysis.columns and not df_analysis['heures_sommeil'].isna().all():
           axes[1, 1].hist(df_analysis['heures_sommeil'].dropna(), bins=10, alpha=0.7)
           axes[1, 1].set_xlabel('Heures de Sommeil')
           axes[1, 1].set_ylabel('Fréquence')
           axes[1, 1].set_title('Distribution des Heures de Sommeil')
       
       plt.tight_layout()
       return fig

**Graphiques créés :**
1. **Graphique en secteurs** : Distribution de la qualité du sommeil
2. **Graphique linéaire** : Évolution du sentiment dans le temps
3. **Nuage de points** : Corrélation fatigue vs sentiment
4. **Histogramme** : Distribution des heures de sommeil

Fonctions Utilitaires
=====================

Données d'Exemple
------------------

.. code-block:: python

   def load_sample_data():
       """Chargement de données d'exemple"""
       sample_entries = [
           "J'ai très mal dormi cette nuit, je me suis réveillé 4 fois et je suis très fatigué ce matin",
           "Excellente nuit de sommeil, j'ai dormi 8 heures d'affilée et je me sens très reposé",
           "Nuit difficile à cause du stress, j'ai mis du temps à m'endormir et j'ai fait des cauchemars",
           "Sommeil correct, environ 7 heures mais quelques réveils nocturnes",
           "Je suis épuisé, seulement 5 heures de sommeil à cause du travail",
           "Très bonne récupération, je me sens énergique et frais ce matin",
           "Insomnie partielle, réveillé à 3h du matin et impossible de me rendormir",
           "Nuit paisible, endormissement rapide et réveil naturel après 8h30 de sommeil"
       ]
       return sample_entries

**Variété des exemples :**
- Mauvaise qualité avec réveils multiples
- Excellente qualité avec durée optimale
- Problèmes liés au stress
- Qualité moyenne
- Durée insuffisante
- Très bonne récupération
- Insomnie
- Sommeil optimal

Fonction d'Analyse Complète
---------------------------

.. code-block:: python

   def run_complete_analysis(entries):
       """Fonction principale pour l'analyse complète"""
       analyzer = SleepDiaryAnalyzer()
       
       # Analyse des entrées
       df_results = analyzer.analyze_sleep_diary(entries)
       
       # Génération d'insights
       insights = analyzer.generate_insights(df_results)
       
       # Création des visualisations
       fig = analyzer.create_visualizations(df_results)
       
       return df_results, insights, fig

**Processus complet :**
1. Initialisation de l'analyseur
2. Analyse de toutes les entrées du journal
3. Génération des insights globaux
4. Création des visualisations
5. Retour de tous les résultats

Exécution du Programme Principal
================================

Script Principal
----------------

.. code-block:: python

   if __name__ == "__main__":
       # Test avec des données d'exemple
       sample_data = load_sample_data()
       
       print("=== ANALYSE DU JOURNAL DE SOMMEIL ===\n")
       
       # Analyse
       results_df, insights_dict, visualization = run_complete_analysis(sample_data)
       
       # Affichage des résultats
       print("Résultats de l'analyse:")
       print(f"- Nombre d'entrées analysées: {insights_dict['total_entries']}")
       print(f"- Sentiment moyen: {insights_dict['avg_sentiment']:.3f}")
       print(f"- Distribution de qualité: {insights_dict['quality_distribution']}")
       
       if insights_dict['most_common_issues']:
           print(f"- Problèmes les plus fréquents: {insights_dict['most_common_issues']}")
       
       if insights_dict['avg_sleep_hours']:
           print(f"- Durée moyenne de sommeil: {insights_dict['avg_sleep_hours']:.1f}h")
       
       # Sauvegarde des résultats
       results_df.to_csv('sleep_analysis_results.csv', index=False)
       
       if visualization:
           visualization.savefig('sleep_analysis_visualization.png', dpi=300, bbox_inches='tight')
       
       print("\nAnalyse terminée! Fichiers sauvegardés.")

**Étapes d'exécution :**
1. Chargement des données d'exemple
2. Affichage du titre
3. Exécution de l'analyse complète
4. Affichage des résultats principaux
5. Sauvegarde des résultats en CSV
6. Sauvegarde de la visualisation en PNG
7. Message de confirmation

Reproduction des Résultats
===========================

Pour obtenir exactement les mêmes résultats :

1. **Installation des dépendances** :
   
   .. code-block:: bash
   
      pip install pandas numpy nltk scikit-learn matplotlib seaborn

2. **Exécution du code complet** :
   
   - Copiez tout le code dans un fichier Python (ex: `sleep_analyzer.py`)
   - Exécutez : `python sleep_analyzer.py`

3. **Fichiers générés** :
   
   - `sleep_analysis_results.csv` : Résultats détaillés de l'analyse
   - `sleep_analysis_visualization.png` : Graphiques de visualisation

4. **Résultats attendus** :
   
   - 8 entrées analysées
   - Sentiment moyen calculé
   - Distribution de qualité (Bonne/Moyenne/Mauvaise)
   - Problèmes les plus fréquents identifiés
   - Durée moyenne de sommeil calculée

Personnalisation
================

Adaptation à Vos Données
-------------------------

Pour analyser vos propres données de journal de sommeil :

1. **Remplacez les données d'exemple** :
   
   .. code-block:: python
   
      mes_entrees = [
          "Votre première entrée de journal",
          "Votre deuxième entrée de journal",
          # ... plus d'entrées
      ]
      
      results_df, insights_dict, visualization = run_complete_analysis(mes_entrees)

2. **Ajoutez des mots-clés personnalisés** :
   
   Modifiez le dictionnaire `sleep_keywords` pour inclure vos propres termes.

3. **Personnalisez les seuils de classification** :
   
   Ajustez les coefficients dans `classify_sleep_quality()` selon vos besoins.

4. **Ajoutez de nouvelles métriques** :
   
   Étendez `generate_insights()` avec vos propres calculs.

Cette documentation complète vous permet de comprendre et reproduire exactement tous les aspects de l'analyse de journal de sommeil.