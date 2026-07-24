# BBC Persian — analyse de la première page

Ce programme traite uniquement la première page de :

https://www.bbc.com/persian/topics/cvjp23v3083t

La méthode reste identique :

1. scraping avec Selenium et Beautiful Soup ;
2. traduction du persan vers le français avec NLLB ;
3. analyse positive ou négative avec mDeBERTa XNLI ;
4. export des résultats en CSV et PDF.

## Exécution sous Windows

```bat
.venv\Scripts\activate
python -m pip install -r requirements.txt
python analyse_bbc_1_page.py
```

## Fichiers produits

```text
resultats/articles_traduits.csv
resultats/sentiments_bbc_persian_1_page.csv
resultats/sentiments_bbc_persian_1_page.pdf
```

Le fichier `articles_traduits.csv` sert de sauvegarde intermédiaire afin de
pouvoir reprendre l'analyse sans retraduire les articles déjà traités.
