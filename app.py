#!/usr/bin/env python3
"""Semantic Extractor MCP Server (Python)

A minimalistic MCP server implementation to test functionality.
"""

import logging
import sys
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv
import os
from pathlib import Path
import re
from typing import Any, Dict, List
import requests
import numpy as np
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from transformers import pipeline

# Load environment variables
load_dotenv()

# Setup logging first
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Get Dandelion API token
DANDELION_TOKEN = os.getenv("DANDELION_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Configuration de l'API Dandelion
DANDELION_BASE_URL = "https://api.dandelion.eu/datatxt/nex/v1/"
DANDELION_PARAMS = {
    "min_confidence": 0.5,
    "include": "types,categories,abstract,alternate_labels",
    "lang": "fr"
}

# Vérifier le token Dandelion
if DANDELION_TOKEN:
    logger.info(f"Token Dandelion configuré: {DANDELION_TOKEN[:5]}...")
else:
    logger.warning("ATTENTION : Aucun token Dandelion trouvé ! L'extraction d'entités ne fonctionnera pas.")

# Try to import FastMCP with fallbacks
try:
    from fastmcp import FastMCP
    logger.info("Successfully imported FastMCP")
except ImportError as e:
    logger.error(f"Error importing FastMCP: {e}")
    try:
        from mcp.server.server import Server as FastMCP
        logger.info("Falling back to mcp.server.server.Server")
    except ImportError as e:
        logger.error(f"Failed to import Server class: {e}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# NLTK stop‑words : téléchargement automatique si nécessaire
# ---------------------------------------------------------------------------
import nltk
from nltk.corpus import stopwords as nltk_stopwords

try:
    nltk_stopwords.words("french")
except LookupError:
    logging.info("Téléchargement automatique des stop‑words NLTK…")
    nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Configuration globale
# ---------------------------------------------------------------------------

load_dotenv()
DANDELION_TOKEN: str | None = os.getenv("DANDELION_TOKEN")
DANDELION_BASE_URL = "https://api.dandelion.eu/datatxt"
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("semantic-extractor")

# ---------------------------------------------------------------------------
# Stop‑words FR (NLTK + éventuel fichier custom)
# ---------------------------------------------------------------------------

FRENCH_STOPWORDS: set[str] = set(nltk_stopwords.words("french"))
CUSTOM_STOPWORDS_PATH = Path(__file__).with_name("stopwords_fr.txt")
if CUSTOM_STOPWORDS_PATH.exists():
    with CUSTOM_STOPWORDS_PATH.open(encoding="utf-8") as fh:
        extra = {w.strip().lower() for w in fh if w.strip()}
        FRENCH_STOPWORDS.update(extra)
        LOGGER.info("Loaded %s custom stop‑words", len(extra))
LOGGER.info("Total stop‑words FR : %s", len(FRENCH_STOPWORDS))

# ---------------------------------------------------------------------------
# Regex & constantes NLP
# ---------------------------------------------------------------------------

TOKEN_REGEX_FR = re.compile(r"\b[\wàâäéèêëîïôöùûüç'\-]+\b", re.I | re.U)
WEIGHTS: dict[str, int] = {
    "title": 5,
    "h1": 4,
    "h2": 3,
    "a": 2,
    "strong": 2,
    "first_500_words": 2,
    "default": 1,
}
MAX_DOCUMENT_KEYWORDS = 100

# ---------------------------------------------------------------------------
# Modèle d'embedding SentenceTransformers (chargé lazy)
# ---------------------------------------------------------------------------

from transformers import pipeline

_EXTRACTOR = None

def get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        LOGGER.info("Loading embedding model %s…", MODEL_NAME)
        _EXTRACTOR = pipeline("feature-extraction", MODEL_NAME, device="cpu")
    return _EXTRACTOR


def embed(text):
    emb = get_extractor()(text, truncation=True, max_length=128)[0]
    emb = np.asarray(emb).mean(axis=0)
    emb /= np.linalg.norm(emb) + 1e-9
    return emb.astype(np.float32)


def cosine_sim(a, b):
    return float(np.dot(a, b))

# ---------------------------------------------------------------------------
# Scraping & keyword extraction helpers
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; semantic-extractor/1.0; +https://example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr,fr-FR;q=0.9,en;q=0.8",
}

def fetch_url_text(url):
    LOGGER.info("Fetching %s", url)
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as exc:
        LOGGER.error("Failed to fetch %s : %s", url, exc)
        return ""
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form", "noscript"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body
    text = root.get_text(" ", strip=True) if root else ""
    return " ".join(text.split())


def tokenize_and_weight(text, weight, scores):
    for token in TOKEN_REGEX_FR.findall(text.lower()):
        if len(token) <= 2 or token in FRENCH_STOPWORDS:
            continue
        scores[token] = scores.get(token, 0) + weight


def extract_keywords_from_html(html):
    soup = BeautifulSoup(html, "lxml")
    scores: Dict[str, int] = {}

    tokenize_and_weight(soup.get_text(" ", strip=True), WEIGHTS["default"], scores)
    tokenize_and_weight(soup.title.string if soup.title else "", WEIGHTS["title"], scores)
    if h1 := soup.find("h1"):
        tokenize_and_weight(h1.get_text(), WEIGHTS["h1"], scores)
    for h2 in soup.find_all("h2"):
        tokenize_and_weight(h2.get_text(), WEIGHTS["h2"], scores)
    for a in soup.find_all("a"):
        tokenize_and_weight(a.get_text(), WEIGHTS["a"], scores)
    for strong in soup.find_all("strong"):
        tokenize_and_weight(strong.get_text(), WEIGHTS["strong"], scores)
    words = TOKEN_REGEX_FR.findall(soup.get_text(" ", strip=True).lower())
    tokenize_and_weight(" ".join(words[:500]), WEIGHTS["first_500_words"], scores)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:MAX_DOCUMENT_KEYWORDS]
    return [{"term": t, "score": s} for t, s in ranked]

# ---------------------------------------------------------------------------
# Dandelion entity extraction helper
# ---------------------------------------------------------------------------

DANDELION_PARAMS = {
    "min_length": 2,
    "social.mention": False,
    "social.hashtag": False,
    "country": -1,
    "include": "types,categories,abstract,alternate_labels",
}

def extract_entities_dandelion(text):
    if not DANDELION_TOKEN:
        return []
    try:
        r = requests.get(
            f"{DANDELION_BASE_URL}/nex/v1/",
            params={**DANDELION_PARAMS, "text": text, "token": DANDELION_TOKEN, "lang": "fr"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("annotations", [])
    except Exception as exc:
        LOGGER.error("Dandelion error : %s", exc)
        return []

# ---------------------------------------------------------------------------
# MCP Server & tools
# ---------------------------------------------------------------------------

# Create MCP server instance
mcp = FastMCP("semantic-extractor-py", protocol_version="2024-11-05")

@mcp.tool()
def hello(name=None):
    """Simple test tool to verify MCP functionality."""
    if name:
        return {"message": f"Hello, {name}!", "timestamp": datetime.utcnow().isoformat()}
    return {"message": "Hello, world!", "timestamp": datetime.utcnow().isoformat()}

@mcp.tool()
def get_heartbeat(user=None):
    """Return a simple heartbeat response."""
    user_str = user if user else "unknown user"
    return {
        "message": f"Heartbeat for {user_str}", 
        "timestamp": datetime.utcnow().isoformat()
    }

@mcp.tool()
def echo(text=None):
    """Echo back the input text."""
    return {
        "input": text,
        "timestamp": datetime.utcnow().isoformat()
    }


@mcp.tool()
async def extract_terms_from_url(url, keyword="", ctx=None):
    """Extrait les termes et entités d'une URL et calcule leur similarité avec le mot-clé.
    
    Args:
        url: URL à analyser
        keyword: Mot-clé pour calculer la similarité (optionnel)
        ctx: Contexte (optionnel)
    """
    try:
        # Fetch content
        logger.info(f"Fetching content from {url}")
        response = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; semantic-extractor/1.0)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "fr,fr-FR;q=0.9,en;q=0.8"
        })
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Supprimer les balises non pertinentes pour l'extraction des termes
        for element in soup.find_all(["script", "style", "header", "nav", "footer", "aside", "meta", "noscript"]):
            element.extract()
            
        # Essayer d'extraire le contenu principal
        main_content = soup.find("main") or soup.find("article") or soup.find("div", {"id": "content"}) or soup.find("div", {"class": "content"})
        
        # Si on trouve un contenu principal, l'utiliser, sinon prendre tout le body
        if main_content:
            logger.info("Contenu principal détecté, extraction ciblée.")
            text = main_content.get_text(" ", strip=True)
        else:
            # Si pas de contenu principal identifié, prendre tout le body en excluant les parties non pertinentes
            text = soup.body.get_text(" ", strip=True) if soup.body else soup.get_text(" ", strip=True)
        
        # Extract title
        title = soup.title.string if soup.title else ""
        
        # Chargement des stop-words français depuis un fichier
        def load_stopwords(filepath="stopwords_fr.txt"):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    stopwords = set(word.strip().lower() for word in file.readlines() if word.strip())
                logger.info(f"Total stop‑words FR : {len(stopwords)}")
                return stopwords
            except Exception as e:
                logger.warning(f"Impossible de charger les stop-words depuis {filepath}: {e}")
                # Fallback sur une liste minime si le fichier n'est pas disponible
                return set([
                    "le", "la", "les", "un", "une", "des", "et", "en", "que", "qui", "dans", "sur", "pour",
                    "par", "avec", "sans", "ne", "pas", "plus", "moins", "autre", "cette", "ces", "son", "sa", "ses",
                    "ce", "c'est", "j'ai", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on", "se"  
                ])

        # Chargement des stop-words
        FRENCH_STOP_WORDS = load_stopwords("stopwords_fr.txt")

        # Fonctions d'aide pour le nettoyage et l'extraction de texte
        def clean_text_for_tokens(raw_text):
            # Remplacer apostrophes par espace
            text = re.sub(r"['''‚‛`´]", " ", raw_text)
            # Supprimer la ponctuation
            text = re.sub(r'[\.,;:!?«»()\[\]{}"\/\\&$€%*|=+<>]+', ' ', text)
            # Normaliser les espaces
            text = re.sub(r'\s+', ' ', text).strip().lower()
            return text
            
        def extract_tokens(raw_text, weight=1.0):
            tokens = []
            text = clean_text_for_tokens(raw_text)
            for token in re.findall(r'\b[\w\-]+\b', text):
                # Filtrage: longueur > 1, pas un stopword, contient une lettre
                if (len(token) > 1 and 
                    token not in FRENCH_STOP_WORDS and 
                    re.search(r'[a-zàâäéèêëîïôöùûüÿçñ]', token)):
                    tokens.append(token)
            return tokens
            
        # Dictionnaire pour stocker les termes avec leur pondération
        term_weights = {}
        
        # Appliquer une pondération basée sur l'emplacement des termes
        # 1. Title - Pondération 5.0
        if soup.title and soup.title.string:
            logger.info("Extraction depuis le titre")
            for token in extract_tokens(soup.title.string):
                term_weights[token] = term_weights.get(token, {"count": 0, "weight": 0})
                term_weights[token]["count"] += 1
                term_weights[token]["weight"] += 5.0
                
        # 2. H1 - Pondération 4.0
        for h1 in soup.find_all('h1'):
            for token in extract_tokens(h1.get_text()):
                term_weights[token] = term_weights.get(token, {"count": 0, "weight": 0})
                term_weights[token]["count"] += 1
                term_weights[token]["weight"] += 4.0
                
        # 3. H2 - Pondération 3.0
        for h2 in soup.find_all('h2'):
            for token in extract_tokens(h2.get_text()):
                term_weights[token] = term_weights.get(token, {"count": 0, "weight": 0})
                term_weights[token]["count"] += 1
                term_weights[token]["weight"] += 3.0
                
        # 4. Ancres de liens - Pondération 2.5
        for link in soup.find_all('a'):
            if link.get_text().strip():
                for token in extract_tokens(link.get_text()):
                    term_weights[token] = term_weights.get(token, {"count": 0, "weight": 0})
                    term_weights[token]["count"] += 1
                    term_weights[token]["weight"] += 2.5
                    
        # 5. H3 - Pondération 2.0
        for h3 in soup.find_all('h3'):
            for token in extract_tokens(h3.get_text()):
                term_weights[token] = term_weights.get(token, {"count": 0, "weight": 0})
                term_weights[token]["count"] += 1
                term_weights[token]["weight"] += 2.0
                
        # 6. Strong/Bold - Pondération 1.5
        for strong in soup.find_all(['strong', 'b']):
            for token in extract_tokens(strong.get_text()):
                term_weights[token] = term_weights.get(token, {"count": 0, "weight": 0})
                term_weights[token]["count"] += 1
                term_weights[token]["weight"] += 1.5
        
        # 7. Contenu principal
        clean_text = clean_text_for_tokens(text)
        content_tokens = re.findall(r'\b[\w\-]+\b', clean_text)
        filtered_content = [token for token in content_tokens 
                          if len(token) > 1 
                          and token not in FRENCH_STOP_WORDS 
                          and re.search(r'[a-zàâäéèêëîïôöùûüÿçñ]', token)]
                          
        # Les 500 premiers mots avec pondération 1.2
        first_500 = filtered_content[:500] if len(filtered_content) > 500 else filtered_content
        for token in first_500:
            term_weights[token] = term_weights.get(token, {"count": 0, "weight": 0})
            term_weights[token]["count"] += 1
            term_weights[token]["weight"] += 1.2
            
        # Le reste du contenu avec pondération 1.0
        if len(filtered_content) > 500:
            for token in filtered_content[500:]:
                term_weights[token] = term_weights.get(token, {"count": 0, "weight": 0})
                term_weights[token]["count"] += 1
                term_weights[token]["weight"] += 1.0
                
        # Convertir en liste pour tri
        weighted_terms = [(term, data["count"], data["weight"]) 
                        for term, data in term_weights.items()]
        
        # Tri par poids et par fréquence
        top_terms = sorted(weighted_terms, key=lambda x: (x[2], x[1]), reverse=True)[:20]
        
        # Calculate semantic similarity if keyword is provided
        ranked_terms = []
            
        if keyword and keyword.strip():
            logger.info(f"Calcul de similarité pour le mot-clé: {keyword}")
            # Compute embedding for the keyword
            try:
                keyword_embedding = embed(keyword)
                
                # Calculate similarity for each weighted term
                similarities = []
                # On utilise les termes avec leur pondération
                for term, count, weight in top_terms:
                    try:
                        term_embedding = embed(term)
                        similarity = float(cosine_sim(keyword_embedding, term_embedding))
                        # Stockage du terme avec sa fréquence, sa pondération et sa similarité
                        similarities.append({
                            "term": term, 
                            "count": count, 
                            "weight": round(weight, 2),
                            "similarity": round(similarity, 4)
                        })
                    except Exception as e:
                        logger.error(f"Error calculating similarity for term {term}: {e}")
                        similarities.append({
                            "term": term, 
                            "count": count, 
                            "weight": round(weight, 2),
                            "similarity": 0.0
                        })
                
                # Sort by similarity
                ranked_terms = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:20]
            except Exception as e:
                logger.error(f"Error calculating keyword embedding: {e}")
                # Fallback sur le tri par poids/fréquence
                ranked_terms = [{
                    "term": term, 
                    "count": count, 
                    "weight": round(weight, 2),
                    "similarity": 0.0
                } for term, count, weight in top_terms]
        else:
            # Sans mot-clé, on classe par poids puis fréquence
            ranked_terms = [{
                "term": term, 
                "count": count, 
                "weight": round(weight, 2),
                "similarity": 0.0
            } for term, count, weight in top_terms]
        
        # Extract entities using Dandelion API
        entities = []
        if DANDELION_TOKEN:
            try:
                # Limit text length for API call (max 8K pour Dandelion)
                api_text = text[:8000]  # Limit to 8000 chars for API
                
                # Ensure we have some valid text to analyze
                if not api_text or len(api_text.strip()) < 100:
                    logger.warning(f"Texte trop court pour l'analyse d'entités: {len(api_text)} caractères")
                else:
                    logger.info(f"Analyse d'entités sur {len(api_text)} caractères")
                    
                    # Paramètres pour l'API Dandelion
                    params = DANDELION_PARAMS.copy()
                    params.update({
                        "text": api_text,
                        "token": DANDELION_TOKEN
                    })
                    
                    # Appel à l'API
                    api_url = "https://api.dandelion.eu/datatxt/nex/v1/"
                    logger.info(f"Appel API Dandelion: {api_url}")
                    entity_response = requests.post(
                        api_url,
                        data=params,
                        timeout=20
                    )
                    
                    # Log de la réponse HTTP
                    logger.info(f"Réponse Dandelion - Status: {entity_response.status_code}")
                    
                    # Vérification et traitement de la réponse
                    if entity_response.status_code == 200:
                        entity_data = entity_response.json()
                        annotations = entity_data.get("annotations", [])
                        logger.info(f"Entités trouvées: {len(annotations)}")
                        
                        # Dictionnaire pour dédoublonner par URI
                        unique_entities = {}
                        
                        # Construction des entités
                        for entity in annotations:
                            spot = entity.get("spot")
                            uri = entity.get("uri")
                            
                            # Vérifier que l'entité a un nom et un URI
                            if not spot or not uri:
                                continue
                                
                            # Créer l'objet d'entité
                            entity_info = {
                                "spot": spot,
                                "confidence": entity.get("confidence", 0.0),
                                "uri": uri,
                                "types": entity.get("types", []),
                                "title": entity.get("title", ""),
                                "abstract": entity.get("abstract", "")
                            }
                            
                            # Calcul de similarité si mot-clé fourni
                            if keyword and keyword.strip():
                                try:
                                    entity_embedding = embed(spot)
                                    entity_info["similarity"] = float(cosine_sim(keyword_embedding, entity_embedding))
                                except Exception as e:
                                    logger.error(f"Erreur calcul similarité pour l'entité '{spot}': {e}")
                                    entity_info["similarity"] = 0.0
                            else:
                                entity_info["similarity"] = 0.0
                            
                            # Dédoublonnage - si l'URI existe déjà et a une meilleure confiance/similarité
                            if uri in unique_entities:
                                existing = unique_entities[uri]
                                # Conserver l'entité avec la plus grande confiance OU similarité
                                if keyword and keyword.strip():
                                    if entity_info["similarity"] > existing["similarity"]:
                                        unique_entities[uri] = entity_info
                                else:
                                    if entity_info["confidence"] > existing["confidence"]:
                                        unique_entities[uri] = entity_info
                            else:
                                # Nouvelle entité unique
                                unique_entities[uri] = entity_info
                        
                        # Convertir le dictionnaire en liste pour le tri final
                        entities = list(unique_entities.values())
                        logger.info(f"Entités uniques extraites: {len(entities)}")
                        
                        # Tri des entités par similarité ou confiance
                        if keyword and keyword.strip():
                            entities = sorted(entities, key=lambda x: x["similarity"], reverse=True)[:15]
                        else:
                            entities = sorted(entities, key=lambda x: x.get("confidence", 0), reverse=True)[:15]
                    else:
                        # Erreur API - log complet de la réponse
                        try:
                            error_data = entity_response.json()
                            logger.error(f"Erreur API Dandelion: {error_data}")
                        except Exception:
                            logger.error(f"Erreur API Dandelion: {entity_response.text[:500]}")
            except Exception as e:
                logger.error(f"Exception lors de l'appel à l'API Dandelion: {e}")
        else:
            logger.warning("Extraction d'entités impossible: aucun token Dandelion configuré")
        
        return {
            "url": url,
            "title": title,
            "timestamp": datetime.utcnow().isoformat(),
            "top_keywords": ranked_terms,  # Déjà au format dict avec count, weight et similarity
            "entities": entities,
            "text_length": len(content_tokens),
            "keyword": keyword if keyword else None
        }
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return {
            "url": url,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@mcp.tool()
async def extract_terms_from_urls(urls=None, keyword="", ctx=None):
    """Extrait les termes saillants & entités pour plusieurs URLs.
    
    Args:
        urls: Liste d'URLs à analyser
        keyword: Mot-clé pour calculer la similarité sémantique (optionnel)
        ctx: Contexte optionnel
    """
    # Ensure urls is not None
    if urls is None:
        return {"error": "Missing required 'urls' parameter"}
        
    logger.info(f"Analyse de {len(urls)} URLs avec mot-clé: '{keyword}'")
    
    # Process each URL individually
    results = []
    for u in urls:
        # Use the extract_terms_from_url function to analyze each URL
        try:
            # Passage du mot-clé à la fonction d'extraction
            res = await extract_terms_from_url(u, keyword, ctx)
            results.append(res)
        except Exception as e:
            logger.error(f"Error processing URL {u}: {str(e)}")
            results.append({
                "url": u,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

    # Compile des données de tous les URLs pour agrégation
    combined_terms = {}
    all_entities = []
    
    # Extraire les termes et entités de tous les résultats
    for result in results:
        # Ignorer les résultats avec erreurs
        if "error" in result:
            continue
            
        # Agréger les termes avec leurs pondérations et similarités
        for term_data in result.get("top_keywords", []):
            term = term_data.get("term")
            if not term:
                continue
                
            # Récupérer les données du terme
            count = term_data.get("count", 1)
            weight = term_data.get("weight", 1.0)
            similarity = term_data.get("similarity", 0.0)
            
            # Agréger les données pour ce terme
            if term in combined_terms:
                combined_terms[term]["count"] += count
                combined_terms[term]["weight"] += weight
                # Conserver la meilleure similarité
                if similarity > combined_terms[term]["similarity"]:
                    combined_terms[term]["similarity"] = similarity
            else:
                combined_terms[term] = {
                    "term": term,
                    "count": count,
                    "weight": weight,
                    "similarity": similarity
                }
                
        # Collecter toutes les entités
        if "entities" in result:
            all_entities.extend(result["entities"])
    
    # Dédoublonnage des entités par URI
    unique_entities = {}
    for entity in all_entities:
        uri = entity.get("uri")
        if not uri:
            continue
            
        # Conserver l'entité avec la meilleure similarité/confiance
        if keyword.strip():
            # Si mot-clé fourni, priorité sur la similarité
            if uri not in unique_entities or entity.get("similarity", 0) > unique_entities[uri].get("similarity", 0):
                unique_entities[uri] = entity
        else:
            # Sinon, priorité sur la confiance
            if uri not in unique_entities or entity.get("confidence", 0) > unique_entities[uri].get("confidence", 0):
                unique_entities[uri] = entity
    
    # Conversion en liste et tri final
    entities_list = list(unique_entities.values())
    
    # Tri des entités par similarité ou confiance
    if keyword.strip() and any(e.get("similarity", 0) > 0 for e in entities_list):
        entities_list.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    else:
        entities_list.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    # Conversion de l'agrégation des termes en liste
    terms_list = list(combined_terms.values())
    
    # Tri des termes selon la présence d'un mot-clé
    if keyword.strip():
        # Tri par similarité si mot-clé fourni
        terms_list.sort(key=lambda x: x["similarity"], reverse=True)
    else:
        # Sinon, tri par pondération puis fréquence
        terms_list.sort(key=lambda x: (x["weight"], x["count"]), reverse=True)
    
    # Conserver les 30 meilleurs termes
    top_terms = terms_list[:30]
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "url_count": len(urls),
        "results": results,  # Résultats individuels par URL
        "top_terms_overall": top_terms,  # 30 termes saillants
        "top_entities_overall": entities_list[:15]  # 15 entités uniques
    }

@mcp.tool()
async def extract_content_from_urls(urls=None, ctx=None):
    """Extrait simplement le contenu textuel principal d'une ou plusieurs URLs.
    
    Args:
        urls: Liste d'URLs ou URL unique à analyser
        ctx: Contexte optionnel
    
    Returns:
        Dict contenant le contenu principal extrait pour chaque URL
    """
    # Vérifier si urls est None ou vide
    if not urls:
        return {"error": "Missing required 'urls' parameter"}
        
    # Convertir en liste si une seule URL est fournie
    if isinstance(urls, str):
        urls = [urls]
        
    logger.info(f"Extraction du contenu principal de {len(urls)} URLs")
    
    results = []
    for url in urls:
        try:
            # 1. Fetch content
            logger.info(f"Fetching content from {url}")
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            # 2. Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 3. Nettoyer le HTML
            # Supprimer les balises non pertinentes
            for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'meta', 'noscript', 'iframe']):
                tag.decompose()
            
            # 4. Extraire le contenu principal
            title = soup.title.string if soup.title else ""
            
            # Essayer d'identifier le contenu principal
            main_content = None
            for selector in ['main', 'article', 'div[class*="content"]', 'div[class*="article"]', 'div[id*="content"]', 'div[id*="article"]']:
                content = soup.select(selector)
                if content:
                    main_content = content[0]
                    logger.info(f"Contenu principal détecté via sélecteur: {selector}")
                    break
            
            # Si aucun sélecteur n'a fonctionné, utiliser le body
            if not main_content:
                main_content = soup.body or soup
                logger.info("Pas de contenu principal détecté, utilisation du body")
            
            # Extraire le texte du contenu principal
            content_text = main_content.get_text(" ", strip=True)
            
            # 5. Nettoyer le texte
            # Supprimer les lignes vides
            content_text = re.sub(r'\n\s*\n', '\n', content_text)
            # Normaliser les espaces
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            
            # 6. Collecter les paragraphes 
            paragraphs = []
            for p in main_content.find_all('p'):
                text = p.get_text(strip=True)
                if text and len(text) > 50:  # Ignorer les paragraphes trop courts
                    paragraphs.append(text)
            
            # Ajouter le résultat
            results.append({
                "url": url,
                "title": title,
                "timestamp": datetime.utcnow().isoformat(),
                "content": content_text,
                "paragraphs": paragraphs,
                "content_length": len(content_text)
            })
            
        except Exception as e:
            logger.error(f"Error extracting content from URL {url}: {str(e)}")
            results.append({
                "url": url,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    # Retourner soit une liste de résultats, soit un seul résultat selon l'input
    if len(results) == 1 and isinstance(urls, list) and len(urls) == 1:
        return results[0]
    else:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "url_count": len(urls),
            "results": results
        }

# Main entry point
if __name__ == "__main__":
    logger.info("Starting semantic-extractor-py MCP server...")
    try:
        mcp.run()  # stdio transport by default
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        sys.exit(1)