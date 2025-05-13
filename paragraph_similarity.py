#!/usr/bin/env python3
"""Module de similarité de paragraphes

Ce module permet de découper le contenu de plusieurs URLs en paragraphes,
puis de calculer leur similarité sémantique par rapport à une question posée.
"""

import logging
import re
import numpy as np
import requests
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
try:
    from nltk.corpus import stopwords as nltk_stopwords
    nltk_stopwords.words("french")
except (LookupError, ImportError):
    logger.info("Téléchargement automatique des ressources NLTK...")
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    from nltk.corpus import stopwords as nltk_stopwords

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("paragraph-similarity")

# Constantes
MIN_PARAGRAPH_LENGTH = 100  # Longueur minimale d'un paragraphe (en caractères)
MAX_KEYWORDS_PER_PARAGRAPH = 10  # Nombre maximal de mots-clés à extraire par paragraphe

# Chargement des stop-words français
def load_stopwords():
    """Charge les stop-words français depuis NLTK et le fichier personnalisé"""
    # Stop-words NLTK
    french_stopwords = set(nltk_stopwords.words("french"))
    
    # Stop-words personnalisés
    stopwords_path = Path(__file__).parent / "stopwords_fr.txt"
    if stopwords_path.exists():
        with open(stopwords_path, "r", encoding="utf-8") as f:
            custom_stopwords = {line.strip().lower() for line in f if line.strip()}
            french_stopwords.update(custom_stopwords)
            logger.info(f"Chargé {len(custom_stopwords)} stop-words personnalisés")
    
    logger.info(f"Total stop-words français: {len(french_stopwords)}")
    return french_stopwords

# Chargement des stop-words
FRENCH_STOPWORDS = load_stopwords()
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; semantic-extractor/1.0; +https://example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr,fr-FR;q=0.9,en;q=0.8",
}

# Modèle d'embedding (chargé paresseusement)
_EMBEDDING_MODEL = None

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Charge le modèle d'embedding de façon paresseuse"""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info(f"Chargement du modèle d'embedding {model_name}...")
        _EMBEDDING_MODEL = pipeline("feature-extraction", model_name, device="cpu")
    return _EMBEDDING_MODEL

def compute_embedding(text):
    """Calcule l'embedding d'un texte"""
    emb = get_embedding_model()(text, truncation=True, max_length=512)[0]
    emb = np.asarray(emb).mean(axis=0)
    emb /= np.linalg.norm(emb) + 1e-9
    return emb.astype(np.float32)

def cosine_similarity(a, b):
    """Calcule la similarité cosinus entre deux vecteurs"""
    return float(np.dot(a, b))

def fetch_url_content(url):
    """Récupère le contenu d'une URL"""
    logger.info(f"Récupération du contenu de {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=45)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de {url}: {str(e)}")
        return None

def extract_paragraphs(html_content):
    """Extrait les paragraphes du contenu HTML"""
    if not html_content:
        return []
    
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Nettoyer le HTML des éléments non pertinents
    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'meta', 'iframe']):
        tag.decompose()
    
    # Essayer d'identifier le contenu principal
    main_content = None
    for selector in ['main', 'article', 'div[class*="content"]', 'div[class*="article"]', 'div[id*="content"]', 'div[id*="article"]']:
        content = soup.select(selector)
        if content:
            main_content = content[0]
            break
    
    if not main_content:
        main_content = soup.body or soup
    
    # Extraire le titre si disponible
    title = soup.title.string if soup.title else ""
    
    # Extraire les paragraphes
    paragraphs = []
    
    # 1. Essayer d'abord les balises <p>
    p_tags = main_content.find_all('p')
    for p in p_tags:
        text = p.get_text(strip=True)
        if text and len(text) >= MIN_PARAGRAPH_LENGTH:
            paragraphs.append(text)
    
    # 2. Si peu de paragraphes sont trouvés, essayer les divs et autres conteneurs
    if len(paragraphs) < 3:
        for div in main_content.find_all(['div', 'section']):
            # Éviter les divs qui contiennent d'autres divs (conteneurs)
            if not div.find_all(['div', 'section']):
                text = div.get_text(strip=True)
                if text and len(text) >= MIN_PARAGRAPH_LENGTH and text not in paragraphs:
                    paragraphs.append(text)
    
    # 3. Si toujours pas assez de paragraphes, découper le texte principal
    if len(paragraphs) < 2:
        main_text = main_content.get_text(strip=True)
        # Découper sur les sauts de ligne ou autres marques de paragraphe
        potential_paragraphs = re.split(r'\n+|\r\n+|\. {2,}|\.\t+', main_text)
        for para in potential_paragraphs:
            cleaned_para = para.strip()
            if cleaned_para and len(cleaned_para) >= MIN_PARAGRAPH_LENGTH and cleaned_para not in paragraphs:
                paragraphs.append(cleaned_para)
    
    # 4. Dernier recours : découper en blocs de taille similaire
    if len(paragraphs) < 2 and main_text:
        words = main_text.split()
        chunk_size = 100  # ~100 mots par paragraphe
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk and len(chunk) >= MIN_PARAGRAPH_LENGTH and chunk not in paragraphs:
                paragraphs.append(chunk)
    
    return {
        "title": title,
        "paragraphs": paragraphs
    }

def extract_keywords(text, max_keywords=MAX_KEYWORDS_PER_PARAGRAPH):
    """Extrait les mots-clés saillants d'un texte"""
    # Nettoyage du texte
    text = text.lower()
    
    # Supprimer la ponctuation et les caractères spéciaux
    text = re.sub(r'[\.,;:!?«»()\[\]{}"\'\/\\&$€%*|=+<>_\-]', ' ', text)
    
    # Tokenisation
    try:
        tokens = word_tokenize(text, language='french')
    except Exception as e:
        logger.warning(f"Erreur lors de la tokenisation avec NLTK: {e}")
        # Fallback simple sur les espaces
        tokens = text.split()
    
    # Filtrage des stop-words et des mots courts
    filtered_tokens = [token for token in tokens 
                       if token not in FRENCH_STOPWORDS 
                       and len(token) > 2  # Ignorer les mots trop courts
                       and token.isalpha()]  # Ignorer les tokens qui contiennent des chiffres ou caractères spéciaux
    
    # Comptage des occurrences
    word_counts = Counter(filtered_tokens)
    
    # Extraction des mots-clés les plus fréquents
    keywords = word_counts.most_common(max_keywords)
    
    return [{"term": term, "count": count} for term, count in keywords]

def calculate_paragraph_similarities(paragraphs, question):
    """Calcule la similarité entre chaque paragraphe et la question et extrait les mots-clés"""
    if not paragraphs or not question:
        return []
    
    # Calculer l'embedding de la question
    try:
        question_embedding = compute_embedding(question)
    except Exception as e:
        logger.error(f"Erreur lors du calcul de l'embedding de la question: {str(e)}")
        return []
    
    # Calculer la similarité pour chaque paragraphe
    similarities = []
    for paragraph in paragraphs:
        try:
            # Calculer l'embedding du paragraphe
            paragraph_embedding = compute_embedding(paragraph)
            
            # Calculer la similarité cosinus
            similarity = cosine_similarity(question_embedding, paragraph_embedding)
            
            # Extraire les mots-clés du paragraphe
            keywords = extract_keywords(paragraph)
            
            # Ajouter à la liste des résultats
            similarities.append({
                "paragraph": paragraph,
                "similarity": similarity,
                # Ajouter un résumé court du paragraphe (premiers 150 caractères)
                "summary": paragraph[:150] + "..." if len(paragraph) > 150 else paragraph,
                # Ajouter les mots-clés extraits
                "keywords": keywords
            })
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la similarité pour un paragraphe: {str(e)}")
    
    # Trier par similarité décroissante
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities

async def process_urls_and_compute_similarity(urls, question):
    """Traite une liste d'URLs et calcule la similarité des paragraphes avec la question"""
    if isinstance(urls, str):
        urls = [urls]
    
    if not urls or not question:
        return {
            "error": "Les paramètres 'urls' et 'question' sont obligatoires",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    results = []
    all_paragraphs = []
    
    # Traiter chaque URL
    for url in urls:
        try:
            html_content = fetch_url_content(url)
            if not html_content:
                results.append({
                    "url": url,
                    "error": "Impossible de récupérer le contenu",
                    "timestamp": datetime.utcnow().isoformat()
                })
                continue
            
            # Extraire les paragraphes
            extraction_result = extract_paragraphs(html_content)
            paragraphs = extraction_result.get("paragraphs", [])
            title = extraction_result.get("title", "")
            
            if not paragraphs:
                results.append({
                    "url": url,
                    "error": "Aucun paragraphe extrait",
                    "timestamp": datetime.utcnow().isoformat()
                })
                continue
            
            # Calculer la similarité avec la question
            paragraph_similarities = calculate_paragraph_similarities(paragraphs, question)
            
            # Garder les paragraphes les plus pertinents
            top_similarities = paragraph_similarities[:5]  # 5 paragraphes max par URL
            
            # Ajouter le résultat pour cette URL
            url_result = {
                "url": url,
                "title": title,
                "paragraph_count": len(paragraphs),
                "top_paragraphs": top_similarities,
                "timestamp": datetime.utcnow().isoformat()
            }
            results.append(url_result)
            
            # Ajouter les paragraphes à la liste globale avec leur URL source
            for sim in paragraph_similarities:
                sim["source_url"] = url
                sim["source_title"] = title
                all_paragraphs.append(sim)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'URL {url}: {str(e)}")
            results.append({
                "url": url,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    # Trier tous les paragraphes par similarité (sur toutes les URLs)
    all_paragraphs.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Limiter à 20 paragraphes max au total
    top_paragraphs_overall = all_paragraphs[:20]
    
    # Extraire les mots-clés communs des meilleurs paragraphes
    common_keywords = {}
    for paragraph in top_paragraphs_overall[:5]:  # Utiliser les 5 meilleurs paragraphes pour les mots-clés
        keywords = paragraph.get("keywords", [])
        for kw in keywords:
            term = kw["term"]
            count = kw["count"]
            if term in common_keywords:
                common_keywords[term]["count"] += count
                common_keywords[term]["occurrences"] += 1
            else:
                common_keywords[term] = {"term": term, "count": count, "occurrences": 1}
    
    # Trier les mots-clés communs par nombre total d'occurrences puis par compte
    sorted_keywords = sorted(
        common_keywords.values(), 
        key=lambda x: (x["occurrences"], x["count"]), 
        reverse=True
    )[:15]  # Limiter aux 15 mots-clés les plus pertinents
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "url_count": len(urls),
        "question": question,
        "common_keywords": sorted_keywords,  # Mots-clés communs des meilleurs paragraphes
        "results": results,  # Résultats détaillés par URL
        "top_paragraphs_overall": top_paragraphs_overall  # Top paragraphes toutes URLs confondues
    }
