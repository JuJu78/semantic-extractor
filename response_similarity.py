#!/usr/bin/env python3
"""Module de comparaison de similarité sémantique entre réponses et questions

Ce module calcule la similarité sémantique entre une réponse (générée par un LLM)
et une question, puis compare cette similarité avec celle des paragraphes extraits
d'URLs externes.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Réutilisation des fonctions d'embedding du module de similarité des paragraphes
from paragraph_similarity import get_embedding_model, compute_embedding, cosine_similarity

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("response-similarity")

def calculate_response_similarity(question: str, response: str, reference_paragraphs=None):
    """
    Calcule la similarité sémantique entre une réponse générée et une question,
    puis compare cette similarité avec celles des paragraphes de référence.

    Args:
        question: La question posée
        response: La réponse générée (par Claude ou autre LLM)
        reference_paragraphs: Liste optionnelle de paragraphes de référence avec leurs scores de similarité
                             (tels que retournés par find_relevant_paragraphs)

    Returns:
        Un dictionnaire contenant les scores de similarité et les comparaisons
    """
    if not question or not response:
        return {
            "error": "La question et la réponse sont obligatoires",
            "timestamp": datetime.utcnow().isoformat()
        }

    try:
        # Calculer l'embedding de la question
        question_embedding = compute_embedding(question)
        
        # Calculer l'embedding de la réponse
        response_embedding = compute_embedding(response)
        
        # Calculer la similarité entre la question et la réponse
        similarity_score = cosine_similarity(question_embedding, response_embedding)
        
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "response_similarity": {
                "score": similarity_score,
                "percentage": round(similarity_score * 100, 2)
            }
        }
        
        # Si des paragraphes de référence sont fournis, comparer la réponse avec eux
        if reference_paragraphs and isinstance(reference_paragraphs, list) and len(reference_paragraphs) > 0:
            # Extraire les scores des meilleurs paragraphes
            top_paragraph_scores = []
            
            for paragraph in reference_paragraphs:
                if isinstance(paragraph, dict) and "similarity" in paragraph:
                    top_paragraph_scores.append({
                        "similarity": paragraph["similarity"],
                        "percentage": round(paragraph["similarity"] * 100, 2),
                        "source_url": paragraph.get("source_url", ""),
                        "source_title": paragraph.get("source_title", "")
                    })
            
            # Trier par similarité décroissante (juste pour être sûr)
            top_paragraph_scores.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Prendre les 3 meilleurs scores
            top_3_scores = top_paragraph_scores[:3] if len(top_paragraph_scores) >= 3 else top_paragraph_scores
            
            # Calculer la moyenne des scores de similarité des meilleurs paragraphes
            if top_paragraph_scores:
                avg_similarity = sum(p["similarity"] for p in top_paragraph_scores) / len(top_paragraph_scores)
                avg_percentage = round(avg_similarity * 100, 2)
            else:
                avg_similarity = 0
                avg_percentage = 0
            
            # Comparer le score de la réponse avec les meilleurs paragraphes
            comparison = {
                "response_vs_top_paragraphs": {
                    "response_score": similarity_score,
                    "response_percentage": round(similarity_score * 100, 2),
                    "top_paragraphs_avg_score": avg_similarity,
                    "top_paragraphs_avg_percentage": avg_percentage,
                    "difference": round((similarity_score - avg_similarity) * 100, 2),
                    "is_response_better": similarity_score > avg_similarity
                },
                "top_paragraphs": top_3_scores
            }
            
            result["comparison"] = comparison
            
            # Ajouter une interprétation des résultats
            if similarity_score > avg_similarity * 1.1:  # Au moins 10% meilleur
                interpretation = "La réponse générée est significativement plus pertinente que les paragraphes extraits des sources."
            elif similarity_score > avg_similarity:
                interpretation = "La réponse générée est légèrement plus pertinente que les paragraphes extraits des sources."
            elif similarity_score > avg_similarity * 0.9:  # Moins de 10% moins bon
                interpretation = "La réponse générée est presque aussi pertinente que les paragraphes extraits des sources."
            else:
                interpretation = "Les paragraphes extraits des sources sont plus pertinents que la réponse générée."
            
            result["interpretation"] = interpretation
        
        return result
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la similarité: {str(e)}")
        return {
            "error": f"Une erreur est survenue: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }

async def compare_response_to_question(question: str, response: str, reference_paragraphs=None):
    """
    Fonction asynchrone wrapper pour l'outil MCP.

    Args:
        question: La question posée
        response: La réponse générée
        reference_paragraphs: Paragraphes de référence (optionnel)

    Returns:
        Les résultats de similarité et comparaison
    """
    return calculate_response_similarity(question, response, reference_paragraphs)
