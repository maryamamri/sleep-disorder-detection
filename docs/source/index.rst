Prédiction des Troubles du Sommeil à partir de Signaux Biomédicaux
==================================================================

Ce projet permet de prédire les troubles du sommeil en analysant des signaux biomédicaux 
(EEG, ECG, respiration, etc.) grâce à des modèles d'apprentissage automatique.

.. image:: _static/images/Overview.png
   :width: 600px
   :align: center

Pipeline Complet
================

Notre solution suit un pipeline en 4 étapes principales :

1. **Préparation des données** - Nettoyage et préprocessing des signaux
2. **Création et entrainement du modèle** - Architecture et conception du modèle prédictif
4. **Calcul des troubles** - Prédiction et interprétation des résultats

.. toctree::
   :maxdepth: 2
   :caption: Guide Utilisateur:

   data_preparation
   model
   sleep_disorder_calculation
   sleep_app
   NLP

.. toctree::
   :maxdepth: 2
   :caption: Référence:

   api_reference
   troubleshooting

