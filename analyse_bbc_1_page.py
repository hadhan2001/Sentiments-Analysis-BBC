"""
BBC Persian : scraping et analyse des sentiments de la première page.

Méthode conservée :
- Selenium pour ouvrir les pages ;
- Beautiful Soup pour extraire les articles ;
- NLLB pour traduire le persan vers le français ;
- mDeBERTa XNLI pour classer la situation comme positive ou négative ;
- export final en CSV et PDF.

Exécution :
    python analyse_bbc_10_pages.py
"""

import gc
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import torch
from bs4 import BeautifulSoup
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


TOPIC_URL = "https://www.bbc.com/persian/topics/cvjp23v3083t"
PAGES = 1
DOSSIER = Path("resultats")
CSV_TRADUCTIONS = DOSSIER / "articles_traduits.csv"
CSV_FINAL = DOSSIER / "sentiments_bbc_persian_1_page.csv"
PDF_FINAL = DOSSIER / "sentiments_bbc_persian_1_page.pdf"


def nettoyer(texte):
    if not texte:
        return ""
    texte = texte.replace("\xa0", " ")
    texte = re.sub(r"[\t\r\f\v]+", " ", texte)
    texte = re.sub(r" +", " ", texte)
    texte = re.sub(r"\n\s*\n+", "\n", texte)
    return texte.strip()


def decouper(texte, limite=750):
    phrases = re.split(r"(?<=[.!?؟])\s+|\n+", texte)
    morceaux, courant = [], ""

    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue

        if len(courant) + len(phrase) + 1 <= limite:
            courant = f"{courant} {phrase}".strip()
        else:
            if courant:
                morceaux.append(courant)
            courant = phrase

    if courant:
        morceaux.append(courant)

    return morceaux


def navigateur():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,1000")
    options.add_argument("--lang=fa")
    return webdriver.Chrome(options=options)


def attendre(driver):
    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    time.sleep(1)


def liens_des_10_pages(driver):
    liens = {}

    for page in range(1, PAGES + 1):
        url = TOPIC_URL if page == 1 else f"{TOPIC_URL}?page={page}"
        print(f"Page {page}/{PAGES} : {url}")

        driver.get(url)
        attendre(driver)

        soup = BeautifulSoup(driver.page_source, "lxml")
        zone = soup.find("main") or soup
        avant = len(liens)

        for balise in zone.find_all("a", href=True):
            adresse = urljoin(driver.current_url, balise["href"])
            chemin = urlparse(adresse).path.rstrip("/")

            if re.fullmatch(r"/persian/articles/[A-Za-z0-9]+", chemin):
                liens.setdefault(adresse.split("?")[0], page)

        print(
            f"  Nouveaux articles : {len(liens) - avant} | "
            f"Total : {len(liens)}"
        )

    return liens


def extraire_article(driver, url):
    driver.get(url)
    attendre(driver)

    soup = BeautifulSoup(driver.page_source, "lxml")
    zone = soup.find("article") or soup.find("main")

    if zone is None:
        raise RuntimeError("Contenu principal introuvable.")

    h1 = zone.find("h1")
    titre = nettoyer(h1.get_text(" ", strip=True) if h1 else "")

    paragraphes = [
        nettoyer(p.get_text(" ", strip=True))
        for p in zone.select("[data-component='text-block'] p")
    ]

    if not paragraphes:
        paragraphes = []

        for p in zone.find_all("p"):
            if p.find_parent(["nav", "footer", "aside", "figure"]):
                continue

            texte = nettoyer(p.get_text(" ", strip=True))

            if len(texte) >= 30:
                paragraphes.append(texte)

    uniques = []
    deja_vus = set()

    for paragraphe in paragraphes:
        if paragraphe and paragraphe not in deja_vus and len(paragraphe) >= 30:
            deja_vus.add(paragraphe)
            uniques.append(paragraphe)

    texte = "\n".join(uniques)

    if len(texte) < 100:
        raise RuntimeError("Article vide ou trop court.")

    return titre, texte


def traduire_articles(articles):
    if CSV_TRADUCTIONS.exists():
        deja = pd.read_csv(CSV_TRADUCTIONS, encoding="utf-8-sig")
        termines = set(deja["url"].dropna())
        lignes = deja.to_dict("records")
    else:
        termines = set()
        lignes = []

    nom = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(nom, src_lang="pes_Arab")
    modele = AutoModelForSeq2SeqLM.from_pretrained(nom)
    modele.eval()

    for numero, article in enumerate(articles, start=1):
        if article["url"] in termines:
            continue

        print(
            f"Traduction {numero}/{len(articles)} : "
            f"{article['titre_persan'][:70]}"
        )

        morceaux_fr = []

        for morceau in decouper(
            article["titre_persan"] + "\n" + article["texte_persan"]
        ):
            entree = tokenizer(
                morceau,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            with torch.inference_mode():
                sortie = modele.generate(
                    **entree,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"),
                    max_length=512,
                    num_beams=3,
                )

            morceaux_fr.append(
                tokenizer.decode(sortie[0], skip_special_tokens=True)
            )

        article["texte_francais"] = "\n".join(morceaux_fr)
        lignes.append(article)

        pd.DataFrame(lignes).to_csv(
            CSV_TRADUCTIONS,
            index=False,
            encoding="utf-8-sig",
        )

    del modele
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(lignes)


def analyser_sentiments(tableau):
    if CSV_FINAL.exists():
        final = pd.read_csv(CSV_FINAL, encoding="utf-8-sig")
        termines = set(final["url"].dropna())
        resultats = final.to_dict("records")
    else:
        termines = set()
        resultats = []

    analyseur = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=0 if torch.cuda.is_available() else -1,
    )

    labels = ["une situation négative", "une situation positive"]

    for numero, article in tableau.iterrows():
        if article["url"] in termines:
            continue

        print(
            f"Sentiment {numero + 1}/{len(tableau)} : "
            f"{str(article['titre_persan'])[:70]}"
        )

        negatif = 0.0
        positif = 0.0
        poids_total = 0

        for morceau in decouper(str(article["texte_francais"])):
            resultat = analyseur(
                morceau,
                candidate_labels=labels,
                hypothesis_template="Ce texte décrit {}.",
                multi_label=False,
            )

            scores = dict(zip(resultat["labels"], resultat["scores"]))
            poids = max(len(morceau), 1)
            negatif += scores.get("une situation négative", 0.0) * poids
            positif += scores.get("une situation positive", 0.0) * poids
            poids_total += poids

        if poids_total == 0:
            sentiment = "ERREUR"
            score = 0.0
        else:
            negatif /= poids_total
            positif /= poids_total

            if negatif >= positif:
                sentiment = "NÉGATIF"
                score = negatif
            else:
                sentiment = "POSITIF"
                score = positif

        resultats.append(
            {
                "page": article["page"],
                "url": article["url"],
                "titre_persan": article["titre_persan"],
                "sentiment": sentiment,
                "score": round(score, 4),
                "date_analyse": datetime.now().isoformat(timespec="seconds"),
            }
        )

        pd.DataFrame(resultats).to_csv(
            CSV_FINAL,
            index=False,
            encoding="utf-8-sig",
        )

    return pd.DataFrame(resultats)


def creer_pdf(tableau):
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(PDF_FINAL),
        pagesize=landscape(A4),
        rightMargin=1 * cm,
        leftMargin=1 * cm,
        topMargin=1 * cm,
        bottomMargin=1 * cm,
    )

    comptes = tableau["sentiment"].value_counts()
    elements = [
        Paragraph("Analyse des sentiments - BBC Persian", styles["Title"]),
        Spacer(1, 0.4 * cm),
        Paragraph(
            f"Rubrique : Afghanistan | Pages : 1 à {PAGES} | "
            f"Articles : {len(tableau)}",
            styles["Normal"],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            f"Négatifs : {int(comptes.get('NÉGATIF', 0))} | "
            f"Positifs : {int(comptes.get('POSITIF', 0))}",
            styles["Heading2"],
        ),
        Spacer(1, 0.4 * cm),
    ]

    traductions = pd.read_csv(CSV_TRADUCTIONS, encoding="utf-8-sig")
    titres_fr = {}

    for _, ligne in traductions.iterrows():
        texte_fr = nettoyer(str(ligne.get("texte_francais", "")))
        titres_fr[ligne["url"]] = texte_fr[:130] + (
            "..." if len(texte_fr) > 130 else ""
        )

    donnees = [["Page", "Titre traduit / début", "Sentiment", "Score"]]

    for _, ligne in tableau.iterrows():
        donnees.append(
            [
                str(ligne["page"]),
                Paragraph(
                    titres_fr.get(ligne["url"], ligne["url"]),
                    styles["BodyText"],
                ),
                ligne["sentiment"],
                f"{float(ligne['score']):.2%}",
            ]
        )

    table = Table(
        donnees,
        colWidths=[1.5 * cm, 19 * cm, 3.2 * cm, 2.5 * cm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    elements.append(table)
    elements.append(Spacer(1, 0.4 * cm))
    elements.append(
        Paragraph(
            "Remarque : le résultat décrit la tonalité de la situation "
            "présentée dans l'article, pas nécessairement la position "
            "éditoriale de BBC Persian.",
            styles["Normal"],
        )
    )
    document.build(elements)


def main():
    DOSSIER.mkdir(parents=True, exist_ok=True)
    driver = navigateur()
    articles = []

    try:
        liens = liens_des_10_pages(driver)
        print(f"\nArticles uniques trouvés : {len(liens)}\n")

        for numero, (url, page) in enumerate(liens.items(), start=1):
            print(f"Scraping {numero}/{len(liens)} : {url}")

            try:
                titre, texte = extraire_article(driver, url)
                articles.append(
                    {
                        "page": page,
                        "url": url,
                        "titre_persan": titre,
                        "texte_persan": texte,
                    }
                )
            except Exception as erreur:
                print(
                    f"  Article ignoré : "
                    f"{type(erreur).__name__}: {erreur}"
                )
    finally:
        driver.quit()

    if not articles and not CSV_TRADUCTIONS.exists():
        raise RuntimeError("Aucun article exploitable n'a été trouvé.")

    traductions = traduire_articles(articles)
    resultats = analyser_sentiments(traductions)
    creer_pdf(resultats)

    print("\nAnalyse terminée.")
    print(f"CSV : {CSV_FINAL.resolve()}")
    print(f"PDF : {PDF_FINAL.resolve()}")


if __name__ == "__main__":
    main()
