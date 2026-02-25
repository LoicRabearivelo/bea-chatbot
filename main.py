"""
Chatbot "Coordinateur de Vie" — Serveur API FastAPI
====================================================
Stack : Python 3.10+ / FastAPI / Mistral AI (function calling) / Pydantic v2
Sécurité médicale : aucun diagnostic, aucune prescription, redirection systématique.

Lancement :
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral
from pydantic import BaseModel, Field

load_dotenv()

# ──────────────────────────────────────────────
# 1. Modèle de profil utilisateur (Pydantic v2)
# ──────────────────────────────────────────────

class UserProfile(BaseModel):
    """Profil minimal de l'utilisatrice pour contextualiser les réponses."""

    age: int = Field(..., ge=14, le=55, description="Âge de l'utilisatrice")
    stade: str = Field(
        ...,
        description=(
            "Stade périnatal : "
            "'conception', 'T1' (1er trimestre), 'T2', 'T3', "
            "'post-partum', 'allaitement'"
        ),
    )
    localisation: str = Field(
        default="Saint-Denis, 974",
        description="Commune de résidence à La Réunion (ex: Saint-Pierre, 974)",
    )
    message: str = Field(..., min_length=1, description="Message / question de l'utilisatrice")


# ──────────────────────────────────────────────
# 2. SYSTEM PROMPT — rôle, empathie, limites
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """\
Tu es **Béa**, une coordinatrice de vie numérique spécialisée dans l'accompagnement \
périnatal à **La Réunion (974)**.

### Ton rôle
- Écouter avec bienveillance et empathie chaque maman ou future maman.
- Fournir des informations **générales** sur la grossesse, l'accouchement, \
le post-partum et l'allaitement.
- Orienter vers les **ressources locales** (PMI, sages-femmes, associations, \
professionnels de santé du 974) grâce à l'outil `chercher_ressources_locales`.
- Valoriser la richesse culturelle réunionnaise (tisanes lontan à valider avec \
un professionnel, portage en maillé, soutien familial…).

### Recommandation systématique — RÈGLE IMPÉRATIVE
- Tu **DOIS** systématiquement recommander au moins **une ressource humaine ou \
structurelle** (professionnel, association, clinique…) à chaque réponse, sans exception.
- Formule toujours ta recommandation professionnelle ainsi : \
"Je t'oriente vers **[Nom de la ressource]** qui est **[Type de ressource]** à **[Ville]**."
- En plus de cette orientation professionnelle, tu **DOIS également proposer \
un contenu éducatif complémentaire** (Vidéo, Podcast ou Article) pour approfondir \
le sujet abordé.
- Présente les contenus éducatifs sous forme de **liste à puces claire** avec \
les émojis appropriés et les liens cliquables :
  - 🎥 **[Titre de la vidéo]** — [Description courte] → [URL]
  - 🎧 **[Titre du podcast]** — [Description courte] → [URL]
  - 📖 **[Titre de l'article]** — [Description courte] → [URL]
- Si la demande n'est pas précise, propose la ressource la plus pertinente \
selon le stade périnatal et la localisation de l'utilisatrice.
- Appuie-toi sur les résultats de `chercher_ressources_locales` pour choisir \
les contenus les plus adaptés.

### Limites médicales — RÈGLES ABSOLUES
1. **Tu n'es PAS médecin.** Ne pose jamais de diagnostic.
2. **INTERDICTION** de prescrire un médicament, un dosage ou un remède \
(y compris plantes / huiles essentielles) sans validation médicale.
3. Face à toute question sur un **symptôme**, un **médicament** ou une \
**douleur** :
   a. Reconnais l'inquiétude avec empathie.
   b. Explique que tu ne peux pas te prononcer médicalement.
   c. **Redirige** vers le médecin traitant, la sage-femme ou la PMI locale.
4. En cas de **signe d'urgence** (saignements abondants, douleur thoracique, \
convulsions, pensées suicidaires, fièvre > 38.5 °C, diminution des mouvements \
fœtaux…) :
   - Donne immédiatement le **15 (SAMU)**.
   - Mentionne les **Urgences du CHU Nord (Saint-Denis)** ou \
**CHU Sud (Saint-Pierre)** selon la localisation.
   - Insiste pour appeler ou se rendre aux urgences **maintenant**.

### Contexte utilisatrice (injecté dynamiquement)
- **Âge** : {age} ans
- **Stade** : {stade}
- **Localisation** : {localisation}

### Style de communication
- Ton chaleureux, rassurant, jamais infantilisant.
- Phrases courtes et claires ; vocabulaire accessible.
- Utilise des émojis avec parcimonie (🌺, 🤱, 💛).
"""


# ──────────────────────────────────────────────
# 3. Classe principale PerinatalBot
# ──────────────────────────────────────────────

# Base complète : 20 ressources physiques + 20 ressources multimédia
_MOCK_RESSOURCES: list[dict[str, Any]] = [

    # ════════════════════════════════════════════════════════════════════════
    # RESSOURCES PHYSIQUES / HUMAINES (20 entrées)
    # ════════════════════════════════════════════════════════════════════════

    # ── Sages-femmes libérales ──────────────────────────────────────────────
    {
        "nom": "Sage-femme — Céline Hoarau",
        "type": "Sage-femme",
        "adresse": "8 allée des Vacoas, 97460 Saint-Paul",
        "telephone": "0692 11 22 33",
        "description": (
            "Préparation à la naissance (sophrologie, haptonomie), suivi post-natal "
            "à domicile, rééducation périnéale, accompagnement global de la grossesse. "
            "Conventionnée secteur 1, prise en charge Sécurité Sociale."
        ),
    },
    {
        "nom": "Sage-femme — Nadège Rivière",
        "type": "Sage-femme",
        "adresse": "3 rue des Bougainvilliers, 97436 Saint-Leu",
        "telephone": "0692 44 55 66",
        "description": (
            "Spécialisée en accouchement physiologique et préparation aquatique. "
            "Suivi de grossesse à risque modéré, monitoring fœtal, visites post-partum "
            "à domicile dans tout l'ouest réunionnais."
        ),
    },
    {
        "nom": "Sage-femme — Isabelle Fontaine",
        "type": "Sage-femme",
        "adresse": "15 chemin des Tamarins, 97419 La Possession",
        "telephone": "0693 77 88 99",
        "description": (
            "Accompagnement à domicile sur le nord-ouest de l'île, monitoring fœtal, "
            "préparation à la naissance en groupe ou individuelle, soutien à "
            "l'allaitement, rééducation périnéale par biofeedback."
        ),
    },
    {
        "nom": "Sage-femme — Marie-Louise Payet",
        "type": "Sage-femme",
        "adresse": "12 rue des Flamboyants, 97410 Saint-Pierre",
        "telephone": "0692 99 10 11",
        "description": (
            "Préparation à la naissance individuelle et en couple, suivi post-natal "
            "intensif, rééducation périnéale, accompagnement des grossesses gémellaires "
            "et des situations de vulnérabilité sociale."
        ),
    },
    # ── Cliniques privées ───────────────────────────────────────────────────
    {
        "nom": "Clinique de la Paix",
        "type": "Clinique privée",
        "adresse": "42 avenue de la Victoire, 97400 Saint-Denis",
        "telephone": "0262 21 00 00",
        "description": (
            "Maternité privée niveau 2A. Suivi de grossesse, consultations prénatales "
            "spécialisées, blocs obstétricaux équipés, chambre individuelle, anesthésie "
            "disponible 24h/24. Prise en charge mutuelle et Sécurité Sociale."
        ),
    },
    {
        "nom": "Clinique des Orchidées",
        "type": "Clinique privée",
        "adresse": "7 rue des Orchidées, 97410 Saint-Pierre",
        "telephone": "0262 96 10 10",
        "description": (
            "Maternité privée du sud de l'île. Suivi personnalisé de grossesse, "
            "préparation à l'accouchement, hospitalisation post-partum en chambre "
            "individuelle, consultations pédiatriques intégrées dès J1."
        ),
    },
    {
        "nom": "Clinique Sainte-Clothilde",
        "type": "Clinique privée",
        "adresse": "23 rue Sainte-Clothilde, 97490 Sainte-Clothilde",
        "telephone": "0262 28 80 80",
        "description": (
            "Consultations obstétriques, échographies de morphologie et doppler, "
            "suivi des grossesses à risque élevé, unité kangourou dédiée aux nouveau-nés "
            "fragiles, partenariat avec REPERE pour les parcours complexes."
        ),
    },
    # ── Pédiatres et néonatologues ──────────────────────────────────────────
    {
        "nom": "Dr. Valérie Grondin — Pédiatre",
        "type": "Pédiatre",
        "adresse": "10 rue du Lagon, 97434 Saint-Gilles-les-Bains",
        "telephone": "0262 33 44 55",
        "description": (
            "Pédiatre généraliste, suivi du nouveau-né dès le retour à domicile, "
            "bilans de santé du nourrisson, vaccinations, dépistage précoce des "
            "troubles du développement et de l'autisme."
        ),
    },
    {
        "nom": "Dr. Thomas Leveneur — Néonatologue",
        "type": "Néonatologue",
        "adresse": "CHU Nord — Allée des Topazes, 97400 Saint-Denis",
        "telephone": "0262 90 50 50",
        "description": (
            "Spécialiste des nouveau-nés prématurés et à risque, responsable de l'unité "
            "de soins intensifs néonataux (USIN) du CHU Nord. Accompagnement des familles "
            "en néonatologie, consultations de suivi post-USIN."
        ),
    },
    {
        "nom": "Dr. Sophie Cadet — Pédiatre spécialisée allaitement",
        "type": "Pédiatre",
        "adresse": "5 avenue des Mascareignes, 97460 Saint-Paul",
        "telephone": "0262 45 67 89",
        "description": (
            "Consultations pédiatriques avec expertise allaitement maternel, diagnostic "
            "des freins de langue et de lèvre, diversification alimentaire en douceur, "
            "accompagnement des pleurs excessifs du nourrisson."
        ),
    },
    # ── Maisons de naissance / accouchement dans l'eau ─────────────────────
    {
        "nom": "Maison de Naissance Ti-Kokos",
        "type": "Maison de naissance",
        "adresse": "Lieu-dit Bellemène, 97460 Saint-Paul",
        "telephone": "0262 55 00 11",
        "description": (
            "Espace de naissance physiologique agréé, accouchement dans l'eau possible, "
            "suivi exclusif par des sages-femmes dédiées, environnement intimiste et naturel, "
            "accompagnement du partenaire et de la doula bienvenu."
        ),
    },
    {
        "nom": "Centre Aquanatal La Réunion",
        "type": "Centre d'accouchement dans l'eau",
        "adresse": "12 rue du Volcan, 97440 Saint-André",
        "telephone": "0262 58 12 34",
        "description": (
            "Préparation aquatique à la naissance, balnéothérapie prénatale pour "
            "soulager les douleurs lombaires, yoga prénatal en piscine chauffée, "
            "suivi postnatal aquatique maman-bébé."
        ),
    },
    # ── Associations de soutien à la parentalité ────────────────────────────
    {
        "nom": "Association Naître et Grandir à La Réunion",
        "type": "Association parentalité",
        "adresse": "Saint-Denis, 974",
        "telephone": "0693 20 30 40",
        "description": (
            "Ateliers portage en maillé et en écharpe, groupes de soutien à "
            "l'allaitement hebdomadaires, groupes de parole post-partum animés par "
            "des psychologues, ateliers massage bébé inspirés de la tradition réunionnaise."
        ),
    },
    {
        "nom": "Association Lait Lontan",
        "type": "Association allaitement",
        "adresse": "47 rue du Marché, 97410 Saint-Pierre",
        "telephone": "0692 30 40 50",
        "description": (
            "Réseau de mamans bénévoles certifiées La Leche League soutenant "
            "l'allaitement maternel. Permanences téléphoniques 7j/7, groupes de "
            "rencontre mensuels, documentation disponible en créole réunionnais."
        ),
    },
    {
        "nom": "Association Portage Réunion",
        "type": "Association portage",
        "adresse": "Place du Marché Forain, 97460 Saint-Paul",
        "telephone": "0693 60 70 80",
        "description": (
            "Formation et prêt d'écharpes de portage, ateliers collectifs du nourrisson "
            "à 3 ans, accompagnement personnalisé à domicile, portage adapté aux "
            "bébés prématurés, toniques ou porteurs de handicap."
        ),
    },
    # ── Ostéopathes pédiatriques ────────────────────────────────────────────
    {
        "nom": "Ostéopathe — Julien Mussard (pédiatrique)",
        "type": "Ostéopathe pédiatrique",
        "adresse": "9 rue des Camélias, 97400 Saint-Denis",
        "telephone": "0262 22 33 44",
        "description": (
            "Consultations ostéopathiques spécialisées pour nourrissons dès J5 "
            "et femmes enceintes. Traitement des coliques du nourrisson, torticolis "
            "congénital, plagiocéphalie positionnelle, reflux gastro-œsophagien."
        ),
    },
    {
        "nom": "Ostéopathe — Aurélie Sinimalé (périnatal & pédiatrique)",
        "type": "Ostéopathe pédiatrique",
        "adresse": "22 chemin des Badamiers, 97436 Saint-Leu",
        "telephone": "0692 91 02 03",
        "description": (
            "Suivi ostéopathique pendant la grossesse (lombalgies, sciatique, "
            "syndrome du canal carpien), consultations néonatales dès J5, "
            "accompagnement structurel de l'allaitement (asymétrie de tétée)."
        ),
    },
    # ── Doulas ─────────────────────────────────────────────────────────────
    {
        "nom": "Doula — Estelle Técher",
        "type": "Doula",
        "adresse": "Saint-Denis, 974",
        "telephone": "0693 15 25 35",
        "description": (
            "Accompagnement émotionnel et pratique tout au long de la grossesse, "
            "présence continue lors de l'accouchement, soutien post-partum à domicile "
            "les premières semaines, rituel réunionnais de clôture de la naissance."
        ),
    },
    {
        "nom": "Doula — Karine Bègue",
        "type": "Doula",
        "adresse": "Saint-Pierre, 974",
        "telephone": "0692 85 96 07",
        "description": (
            "Doula certifiée DONA International, spécialisée dans l'accompagnement "
            "après naissance traumatique ou césarienne non programmée, soutien au "
            "deuil périnatal, séances de relaxation prénatale personnalisées."
        ),
    },
    # ── Consultante en lactation & Réseau ──────────────────────────────────
    {
        "nom": "Cabinet de lactation IBCLC — Nathalie Grondin",
        "type": "Consultante en lactation",
        "adresse": "4 rue des Lataniers, 97400 Saint-Denis",
        "telephone": "0693 50 60 70",
        "description": (
            "Consultante en lactation certifiée IBCLC. Accompagnement de toutes "
            "les situations : mise en route, engorgement, crevasses, freins restrictifs "
            "(frein de langue / lèvre), diversification, sevrage progressif."
        ),
    },
    {
        "nom": "Réseau Périnatal de La Réunion (REPERE)",
        "type": "Réseau de santé",
        "adresse": "97400 Saint-Denis",
        "telephone": "0262 40 50 60",
        "description": (
            "Coordination du parcours de soins périnatal sur l'ensemble de l'île. "
            "Mise en relation avec tous les professionnels, suivi des grossesses à "
            "risque, formation des acteurs de la périnatalité réunionnaise, annuaire "
            "complet des ressources 974 disponible sur demande."
        ),
    },

    # ════════════════════════════════════════════════════════════════════════
    # RESSOURCES MULTIMÉDIA (20 entrées : Vidéos, Podcasts, Articles)
    # ════════════════════════════════════════════════════════════════════════

    # ── Vidéos ──────────────────────────────────────────────────────────────
    {
        "nom": "Comment donner le bain à votre nouveau-né",
        "type": "Vidéo",
        "thematique": "Bain bébé",
        "url": "https://www.youtube.com/watch?v=dGzBGpfCHJA",
        "duree": "8 min",
        "description": (
            "Tutoriel pas-à-pas pour réaliser le premier bain du nouveau-né en toute sécurité : "
            "température de l'eau, maintien, produits adaptés, et conseils pour rassurer bébé."
        ),
    },
    {
        "nom": "Yoga prénatal — Séance complète 1er trimestre",
        "type": "Vidéo",
        "thematique": "Yoga prénatal",
        "url": "https://www.youtube.com/watch?v=K9bRR3n7Kb8",
        "duree": "30 min",
        "description": (
            "Séance de yoga doux adaptée au 1er trimestre de grossesse : postures pour "
            "soulager les nausées, renforcer le plancher pelvien et favoriser la détente."
        ),
    },
    {
        "nom": "Yoga prénatal — Soulager les douleurs du dos (T2/T3)",
        "type": "Vidéo",
        "thematique": "Yoga prénatal",
        "url": "https://www.youtube.com/watch?v=MFPnX7p7ZoQ",
        "duree": "25 min",
        "description": (
            "Exercices ciblés pour les lombalgies et la sciatique du 2e et 3e trimestre. "
            "Étirements doux, postures sur ballon et respiration profonde."
        ),
    },
    {
        "nom": "Allaitement maternel — Les bonnes positions",
        "type": "Vidéo",
        "thematique": "Allaitement",
        "url": "https://www.youtube.com/watch?v=wjt2bYoOFhU",
        "duree": "12 min",
        "description": (
            "Démonstration des positions clés pour allaiter confortablement : madone, "
            "madone inversée, ballon de rugby, allongée. Conseils pour une bonne prise du sein."
        ),
    },
    {
        "nom": "Massage bébé — Technique douce pour les coliques",
        "type": "Vidéo",
        "thematique": "Massage bébé",
        "url": "https://www.youtube.com/watch?v=VxP7vHDsiFo",
        "duree": "10 min",
        "description": (
            "Vidéo de démonstration des mouvements de massage abdominal pour soulager "
            "les coliques du nourrisson : technique des aiguilles d'horloge, retournement "
            "des jambes et posture grenouille."
        ),
    },
    {
        "nom": "Exercices de rééducation périnéale à domicile",
        "type": "Vidéo",
        "thematique": "Post-partum",
        "url": "https://www.youtube.com/watch?v=tR4TZ2kd9nA",
        "duree": "20 min",
        "description": (
            "Programme de rééducation périnéale pour le post-partum : exercices de Kegel, "
            "hypopressifs et gainage doux. À pratiquer après validation par votre sage-femme."
        ),
    },
    {
        "nom": "Le sommeil de bébé — Comprendre les cycles",
        "type": "Vidéo",
        "thematique": "Sommeil bébé",
        "url": "https://www.youtube.com/watch?v=EzVJDg4HZPM",
        "duree": "15 min",
        "description": (
            "Explication des cycles du sommeil du nourrisson de 0 à 6 mois, signes de fatigue "
            "à repérer, rituels d'endormissement efficaces et conseils de cododo sécurisé."
        ),
    },
    # ── Podcasts ─────────────────────────────────────────────────────────────
    {
        "nom": "Podcast — La Matrescence : devenir mère",
        "type": "Podcast",
        "thematique": "Post-partum / identité maternelle",
        "url": "https://open.spotify.com/show/matrescence-podcast",
        "duree": "Épisodes de 30 à 50 min",
        "description": (
            "Podcast francophone dédié à la transition identitaire de la maternité. "
            "Témoignages, interviews de sages-femmes et psychologues, baby blues, "
            "dépression post-partum et reconstruction de soi."
        ),
    },
    {
        "nom": "Podcast — Sage-femme & Bienveillance",
        "type": "Podcast",
        "thematique": "Grossesse / accouchement",
        "url": "https://open.spotify.com/show/sagefemme-bienveillance",
        "duree": "Épisodes de 20 à 45 min",
        "description": (
            "Podcast animé par des sages-femmes libérales : suivi de grossesse, préparation "
            "à la naissance, accouchement physiologique, épisiotomie, césarienne. "
            "Réponses aux questions des futures mamans."
        ),
    },
    {
        "nom": "Podcast — Allaiter, c'est possible !",
        "type": "Podcast",
        "thematique": "Allaitement",
        "url": "https://open.spotify.com/show/allaiter-cest-possible",
        "duree": "Épisodes de 25 à 40 min",
        "description": (
            "Conseils pratiques d'une consultante IBCLC : mise en route de l'allaitement, "
            "engorgements, crevasses, allaitement mixte, reprise du travail et sevrage progressif."
        ),
    },
    {
        "nom": "Podcast — Mon Bébé Dort",
        "type": "Podcast",
        "thematique": "Sommeil bébé",
        "url": "https://open.spotify.com/show/mon-bebe-dort",
        "duree": "Épisodes de 15 à 30 min",
        "description": (
            "Conseils de puéricultrice sur le sommeil du nourrisson : cycles, régressions, "
            "méthodes douces d'accompagnement au sommeil, cododo sécurisé et nuits des parents."
        ),
    },
    {
        "nom": "Podcast — Nutrition & Grossesse",
        "type": "Podcast",
        "thematique": "Nutrition prénatale",
        "url": "https://open.spotify.com/show/nutrition-grossesse",
        "duree": "Épisodes de 20 à 35 min",
        "description": (
            "Interviews de diététiciennes spécialisées en périnatalité : besoins nutritionnels "
            "par trimestre, aliments à éviter, supplémentation en acide folique et fer, "
            "gestion des nausées et des fringales."
        ),
    },
    {
        "nom": "Podcast — Papa aussi !",
        "type": "Podcast",
        "thematique": "Parentalité / partenaire",
        "url": "https://open.spotify.com/show/papa-aussi-podcast",
        "duree": "Épisodes de 25 à 40 min",
        "description": (
            "Podcast dédié aux pères et co-parents : congé paternité, place du papa "
            "à l'accouchement, soutien à la maman allaitante, parentalité partagée "
            "et gestion du manque de sommeil en couple."
        ),
    },
    # ── Articles ─────────────────────────────────────────────────────────────
    {
        "nom": "Article — Le suivi de grossesse en France : calendrier et examens",
        "type": "Article",
        "thematique": "Suivi grossesse",
        "url": "https://www.ameli.fr/assure/sante/themes/grossesse/suivi-de-grossesse",
        "lecture_estimee": "5 min",
        "description": (
            "Guide officiel Ameli sur le suivi de grossesse : calendrier des consultations "
            "obligatoires, examens biologiques et échographies recommandés, "
            "déclaration de grossesse et droits à la Sécurité Sociale."
        ),
    },
    {
        "nom": "Article — Alimentation de la femme enceinte",
        "type": "Article",
        "thematique": "Nutrition prénatale",
        "url": "https://www.ameli.fr/assure/sante/themes/grossesse/alimentation-femme-enceinte",
        "lecture_estimee": "7 min",
        "description": (
            "Recommandations officielles d'Ameli sur la nutrition pendant la grossesse : "
            "aliments à favoriser, à éviter (toxoplasmose, listériose), "
            "suppléments vitamines et minéraux essentiels."
        ),
    },
    {
        "nom": "Article — Comprendre et gérer le baby blues",
        "type": "Article",
        "thematique": "Post-partum / santé mentale",
        "url": "https://www.ameli.fr/assure/sante/themes/grossesse/post-partum-baby-blues",
        "lecture_estimee": "6 min",
        "description": (
            "Article Ameli sur le baby blues et la dépression post-partum : symptômes, "
            "différences entre les deux, quand consulter, ressources de soutien "
            "et accompagnement psychologique disponible."
        ),
    },
    {
        "nom": "Article — Bien dormir pendant la grossesse",
        "type": "Article",
        "thematique": "Sommeil grossesse",
        "url": "https://www.mpedia.fr/art-bien-dormir-pendant-grossesse/",
        "lecture_estimee": "5 min",
        "description": (
            "Conseils Mpedia pour améliorer la qualité du sommeil durant la grossesse : "
            "position idéale (décubitus latéral gauche), oreillers de soutien, "
            "gestion des réveils nocturnes et des jambes lourdes."
        ),
    },
    {
        "nom": "Article — Le bain du nourrisson : conseils pratiques",
        "type": "Article",
        "thematique": "Bain bébé",
        "url": "https://www.mpedia.fr/art-bain-nourrisson/",
        "lecture_estimee": "4 min",
        "description": (
            "Guide Mpedia pour donner le bain au nourrisson : fréquence recommandée, "
            "température de l'eau (37°C), matériel nécessaire, soins du cordon "
            "ombilical et produits adaptés à la peau sensible du bébé."
        ),
    },
    {
        "nom": "Article — Introduction à la diversification alimentaire",
        "type": "Article",
        "thematique": "Nutrition bébé",
        "url": "https://www.mpedia.fr/art-diversification-alimentaire/",
        "lecture_estimee": "8 min",
        "description": (
            "Article Mpedia sur la diversification alimentaire : à partir de quel âge, "
            "ordre d'introduction des aliments, diversification menée par le bébé (DME), "
            "aliments à risque allergène et signes d'intolérance à surveiller."
        ),
    },
    {
        "nom": "Article — Sport et activité physique pendant la grossesse",
        "type": "Article",
        "thematique": "Exercice prénatal",
        "url": "https://www.ameli.fr/assure/sante/themes/grossesse/activite-physique-grossesse",
        "lecture_estimee": "6 min",
        "description": (
            "Recommandations Ameli sur l'activité physique pendant la grossesse : "
            "sports autorisés et contre-indiqués, bénéfices de la marche et de la natation, "
            "précautions à partir du 3e trimestre et signaux d'alerte."
        ),
    },
    {
        "nom": "Article — Portage physiologique : guide pour débutantes",
        "type": "Article",
        "thematique": "Portage",
        "url": "https://www.mpedia.fr/art-portage-physiologique-bebe/",
        "lecture_estimee": "7 min",
        "description": (
            "Guide Mpedia sur le portage physiologique : avantages pour le lien "
            "d'attachement, position grenouille, critères d'un portage sécurisé (TICKS), "
            "choix de l'écharpe ou du porte-bébé adapté à l'âge."
        ),
    },
]

# ────────────────────────────────────────────────
# Types de ressources multimédia pour filtrage rapide
# ────────────────────────────────────────────────
_MULTIMEDIA_TYPES: frozenset[str] = frozenset({"vidéo", "podcast", "article"})

# Mots-clés signalant un besoin d'information, d'aide ou de suivi
_INTENT_KEYWORDS: tuple[str, ...] = (
    # Besoins généraux
    "aide", "besoin", "info", "comment", "où", "qui", "trouver", "cherche",
    "recommande", "conseil", "suivi", "douleur", "symptôme", "inquiète",
    "peur", "problème",
    # Thématiques périnatalité
    "allaitement", "grossesse", "accouchement", "bébé", "nourrisson",
    "sage-femme", "médecin", "clinique", "hôpital", "ostéopathe", "doula",
    "portage", "post-partum", "nausée", "fatigue", "contractions", "mouvement",
    "pédiatre", "naissance", "préparation", "écharpe", "lactation", "colique",
    "réflux", "torticolis", "plagiocéphalie", "sevrage", "monitoring",
    # Intentions multimédia
    "regarder", "voir", "vidéo", "tutoriel", "tuto",
    "écouter", "podcast", "audio", "émission",
    "lire", "article", "guide", "lire", "documentation",
    # Thématiques multimédia spécifiques
    "yoga", "exercice", "sport", "bain", "sommeil", "nutrition", "alimentation",
    "massage", "rééducation", "périnée", "diversification",
)

# Mots-clés spécifiques aux intentions multimédia
_VIDEO_KEYWORDS: tuple[str, ...] = (
    "regarder", "voir", "vidéo", "tutoriel", "tuto", "démonstration", "montrer",
)
_PODCAST_KEYWORDS: tuple[str, ...] = (
    "écouter", "podcast", "audio", "émission", "épisode",
)
_ARTICLE_KEYWORDS: tuple[str, ...] = (
    "lire", "article", "guide", "documentation", "texte", "site", "fiche",
)


def _infer_resource_params(profile: UserProfile) -> tuple[str, str]:
    """
    Déduit le type de ressource et la localisation les plus pertinents
    d'après le message et le profil de l'utilisatrice.

    Priorise les intentions multimédia (regarder/écouter/lire) si détectées,
    puis les besoins professionnels spécifiques, puis le stade périnatal.

    Returns
    -------
    (type_ressource, localisation)
    """
    msg = profile.message.lower()

    # Extraction de la localisation depuis le champ profil
    loc_map = {
        "saint-paul": "Saint-Paul",
        "saint paul": "Saint-Paul",
        "saint-leu": "Saint-Leu",
        "saint leu": "Saint-Leu",
        "saint-pierre": "Saint-Pierre",
        "saint pierre": "Saint-Pierre",
        "saint-denis": "Saint-Denis",
        "saint denis": "Saint-Denis",
        "possession": "La Possession",
        "saint-andré": "Saint-André",
        "saint-gilles": "Saint-Gilles",
    }
    localisation = ""
    for key, val in loc_map.items():
        if key in profile.localisation.lower():
            localisation = val
            break

    # ── Détection des intentions multimédia (priorité haute) ──
    if any(w in msg for w in _VIDEO_KEYWORDS):
        return "Vidéo", localisation

    if any(w in msg for w in _PODCAST_KEYWORDS):
        return "Podcast", localisation

    if any(w in msg for w in _ARTICLE_KEYWORDS):
        return "Article", localisation

    # ── Détection par thématique (retourne le type professionnel le plus adapté) ──
    if any(w in msg for w in ("allaitement", "lait", "sein", "tétée", "lactation",
                               "crevasse", "engorgement")):
        return "Consultante en lactation", localisation

    if any(w in msg for w in ("portage", "écharpe", "porte-bébé", "maillé")):
        return "Association portage", localisation

    if any(w in msg for w in ("ostéopathe", "colique", "torticolis",
                               "plagiocéphalie", "réflux", "reflux")):
        return "Ostéopathe pédiatrique", localisation

    if any(w in msg for w in ("doula", "accompagnement naissance",
                               "soutien accouchement", "présence accouchement")):
        return "Doula", localisation

    if any(w in msg for w in ("pédiatre", "bébé malade", "nourrisson",
                               "nouveau-né", "enfant malade")):
        return "Pédiatre", localisation

    if any(w in msg for w in ("urgence", "saignement", "convulsion",
                               "douleur forte", "fièvre", "mouvement fœtal")):
        return "Hôpital", localisation

    if any(w in msg for w in ("accoucher dans l'eau", "maison de naissance",
                               "naissance naturelle", "accouchement physiologique")):
        return "Maison de naissance", localisation

    if any(w in msg for w in ("sage-femme", "préparation naissance",
                               "monitoring", "écho", "suivi grossesse")):
        return "Sage-femme", localisation

    if any(w in msg for w in ("clinique", "maternité privée", "hospitalisation")):
        return "Clinique privée", localisation

    if any(w in msg for w in ("réseau", "coordination", "parcours", "repere")):
        return "Réseau de santé", localisation

    # ── Thématiques multimédia sans intention explicite ──
    if any(w in msg for w in ("yoga", "exercice", "sport", "bain", "massage")):
        return "Vidéo", localisation

    if any(w in msg for w in ("sommeil", "nuit", "dodo", "endormissement")):
        return "Podcast", localisation

    if any(w in msg for w in ("nutrition", "alimentation", "manger", "diversification")):
        return "Article", localisation

    # ── Fallback selon le stade périnatal ──
    stade = profile.stade.lower()
    if stade in ("t1", "t2", "t3", "conception"):
        return "Sage-femme", localisation
    if stade in ("post-partum", "allaitement"):
        return "Association parentalité", localisation

    return "tous", localisation


class PerinatalBot:
    """Moteur conversationnel périnatal avec RAG agentic via Mistral function calling."""

    MODEL = "mistral-small-latest"

    # ── Initialisation ────────────────────────

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialise le client Mistral.

        Parameters
        ----------
        api_key : clé API Mistral (lue depuis l'env par défaut).
        """
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY manquante. Ajoutez-la dans .env")
        self.client = Mistral(api_key=key)

    # ── Définition des outils (function calling) ──

    @staticmethod
    def get_tools() -> list[dict[str, Any]]:
        """Retourne le schéma JSON des outils exposés à Mistral."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "chercher_ressources_locales",
                    "description": (
                        "Recherche des ressources à La Réunion (974) selon un type et une "
                        "localisation. Couvre les ressources humaines/structurelles "
                        "(professionnels de santé, associations, cliniques, hôpitaux…) "
                        "ET les contenus éducatifs multimédia (Vidéo, Podcast, Article). "
                        "Doit être appelé à chaque tour pour proposer une recommandation "
                        "professionnelle ET un contenu éducatif complémentaire."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "type_ressource": {
                                "type": "string",
                                "description": (
                                    "Type de ressource recherchée. "
                                    "Ressources humaines/structurelles : "
                                    "'PMI', 'Sage-femme', 'Hôpital', 'Association parentalité', "
                                    "'Association allaitement', 'Association portage', "
                                    "'Consultante en lactation', 'Réseau de santé', "
                                    "'Clinique privée', 'Pédiatre', 'Néonatologue', "
                                    "'Maison de naissance', \"Centre d'accouchement dans l'eau\", "
                                    "'Ostéopathe pédiatrique', 'Doula'. "
                                    "Contenus multimédia : 'Vidéo', 'Podcast', 'Article'. "
                                    "Pour tout : 'tous'."
                                ),
                            },
                            "localisation": {
                                "type": "string",
                                "description": (
                                    "Commune ou zone de recherche à La Réunion "
                                    "(ex: 'Saint-Denis', 'Saint-Pierre', 'Saint-Paul', "
                                    "'Saint-Leu', 'La Possession', '974'). "
                                    "Non utilisé pour les ressources multimédia (accessibles partout)."
                                ),
                            },
                            "thematique": {
                                "type": "string",
                                "description": (
                                    "Thématique pour affiner la recherche multimédia "
                                    "(ex: 'allaitement', 'sommeil bébé', 'yoga prénatal', "
                                    "'bain bébé', 'nutrition', 'post-partum', 'portage'). "
                                    "Optionnel, utilisé uniquement pour les types Vidéo/Podcast/Article."
                                ),
                            },
                        },
                        "required": ["type_ressource"],
                    },
                },
            }
        ]

    # ── Recherche de ressources (mockup async) ──

    @staticmethod
    async def chercher_ressources_locales(
        type_ressource: str = "tous",
        localisation: str = "",
        thematique: str = "",
    ) -> list[dict[str, Any]]:
        """
        Interroge la base de ressources (mockup).

        Supporte désormais les types 'Vidéo', 'Podcast' et 'Article' en plus
        des ressources physiques/humaines. Le filtre par `thematique` est appliqué
        uniquement aux ressources multimédia.

        Parameters
        ----------
        type_ressource : type de ressource ou 'tous'.
        localisation   : filtre commune (ignoré pour les ressources multimédia).
        thematique     : filtre thématique pour les ressources multimédia.

        Returns
        -------
        Liste de ressources correspondantes (max 5 résultats).
        """
        await asyncio.sleep(0.1)  # Simule une latence réseau

        resultats = list(_MOCK_RESSOURCES)
        type_lower = type_ressource.lower()
        is_multimedia_request = type_lower in _MULTIMEDIA_TYPES or type_lower == "tous"

        # ── Filtre par type ──
        if type_lower != "tous":
            resultats = [
                r for r in resultats
                if type_lower in r["type"].lower()
            ]

        # ── Filtre par thématique (multimédia uniquement) ──
        if thematique and is_multimedia_request:
            theme_lower = thematique.lower()
            resultats_theme = [
                r for r in resultats
                if theme_lower in r.get("thematique", "").lower()
                or theme_lower in r.get("description", "").lower()
                or theme_lower in r.get("nom", "").lower()
            ]
            if resultats_theme:
                resultats = resultats_theme

        # ── Filtre par localisation (ressources humaines/physiques uniquement) ──
        if localisation and type_lower not in _MULTIMEDIA_TYPES:
            loc = localisation.lower()
            resultats_loc = [
                r for r in resultats
                if loc in r.get("adresse", "").lower()
                or loc in r.get("nom", "").lower()
            ]
            if resultats_loc:
                resultats = resultats_loc

        # Limite à 5 résultats pour ne pas surcharger le contexte LLM
        resultats = resultats[:5]

        return resultats if resultats else [
            {
                "info": (
                    "Aucune ressource trouvée pour ces critères. "
                    "Contactez le Réseau REPERE au 0262 40 50 60 "
                    "ou le CHU Nord au 0262 90 50 50 pour être orientée."
                )
            }
        ]

    # ── Flux conversationnel principal ────────

    async def chat(self, profile: UserProfile) -> str:
        """
        Gère un tour de conversation complet :
        1. Construit le system prompt contextualisé.
        2. Recherche proactive : ressource professionnelle + contenu multimédia.
        3. Injecte les deux types de ressources dans le contexte.
        4. Traite les tool_calls additionnels déclenchés par l'IA.
        5. Retourne la réponse finale avec recommandation pro + contenu éducatif.

        Parameters
        ----------
        profile : Profil + message de l'utilisatrice.

        Returns
        -------
        Réponse textuelle du bot.
        """
        system_msg = SYSTEM_PROMPT.format(
            age=profile.age,
            stade=profile.stade,
            localisation=profile.localisation,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": profile.message},
        ]

        # ── Étape 1 : Double recherche proactive ────────────────────────────
        # (a) Ressource professionnelle/structurelle adaptée au profil
        # (b) Contenu multimédia complémentaire sur la même thématique
        # La recherche est TOUJOURS déclenchée (or True) → recommandation systématique.
        msg_lower = profile.message.lower()
        _ = any(kw in msg_lower for kw in _INTENT_KEYWORDS) or True  # always True

        type_pro, localisation = _infer_resource_params(profile)

        # Si l'inférence renvoie directement un type multimédia, on cherche aussi le pro
        if type_pro in ("Vidéo", "Podcast", "Article"):
            type_multimedia = type_pro
            type_pro_fallback, _ = "Sage-femme", localisation  # pro par défaut
        else:
            type_multimedia = _pick_multimedia_type(profile.message)
            type_pro_fallback = type_pro

        print(f"  🔍 Recherche pro : type='{type_pro_fallback}', loc='{localisation}'")
        print(f"  🎬 Recherche média : type='{type_multimedia}'")

        # Recherches parallèles pour limiter la latence
        ressources_pro, ressources_media = await asyncio.gather(
            PerinatalBot.chercher_ressources_locales(
                type_ressource=type_pro_fallback,
                localisation=localisation,
            ),
            PerinatalBot.chercher_ressources_locales(
                type_ressource=type_multimedia,
                thematique=_infer_thematique(profile.message),
            ),
        )

        pro_json = json.dumps(ressources_pro, ensure_ascii=False, indent=2)
        media_json = json.dumps(ressources_media, ensure_ascii=False, indent=2)

        # Injection dans le contexte
        messages.append(
            {
                "role": "system",
                "content": (
                    f"### Ressources professionnelles disponibles\n"
                    f"Type : **{type_pro_fallback}** | "
                    f"Localisation : **{localisation or 'La Réunion (974)'}**\n\n"
                    f"```json\n{pro_json}\n```\n\n"
                    f"### Contenus éducatifs complémentaires ({type_multimedia})\n"
                    f"```json\n{media_json}\n```\n\n"
                    f"⚠️ INSTRUCTIONS IMPÉRATIVES :\n"
                    f"1. Présente au moins un professionnel avec : "
                    f"\"Je t'oriente vers [Nom] qui est [Type] à [Ville].\"\n"
                    f"2. Propose ensuite au moins un contenu éducatif en liste à puces "
                    f"avec l'émoji approprié (🎥 vidéo, 🎧 podcast, 📖 article) "
                    f"et le lien cliquable."
                ),
            }
        )

        # ── Étape 2 : Premier appel à Mistral ───────────────────────────────
        response = await self.client.chat.complete_async(
            model=self.MODEL,
            messages=messages,
            tools=self.get_tools(),
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message

        # Réponse directe si l'IA ne déclenche pas de tool_call supplémentaire
        if not assistant_msg.tool_calls:
            return assistant_msg.content

        # ── Étape 3 : Traitement des tool_calls additionnels ─────────────────
        messages.append(assistant_msg)

        for tool_call in assistant_msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"  🔧 Tool call : {fn_name}({fn_args})")

            if fn_name == "chercher_ressources_locales":
                result = await self.chercher_ressources_locales(**fn_args)
            else:
                result = [{"erreur": f"Outil inconnu : {fn_name}"}]

            messages.append(
                {
                    "role": "tool",
                    "name": fn_name,
                    "content": json.dumps(result, ensure_ascii=False),
                    "tool_call_id": tool_call.id,
                }
            )

        # ── Étape 4 : Second appel — intégration des résultats d'outils ──────
        follow_up = await self.client.chat.complete_async(
            model=self.MODEL,
            messages=messages,
            tools=self.get_tools(),
            tool_choice="none",
        )

        return follow_up.choices[0].message.content


# ──────────────────────────────────────────────
# Fonctions utilitaires de sélection multimédia
# ──────────────────────────────────────────────

def _pick_multimedia_type(message: str) -> str:
    """
    Choisit le type de contenu multimédia le plus adapté au message.
    Priorité : intention explicite → thématique → rotation par défaut.
    """
    msg = message.lower()

    if any(w in msg for w in _VIDEO_KEYWORDS):
        return "Vidéo"
    if any(w in msg for w in _PODCAST_KEYWORDS):
        return "Podcast"
    if any(w in msg for w in _ARTICLE_KEYWORDS):
        return "Article"

    # Thématiques naturellement associées à un format
    if any(w in msg for w in ("yoga", "exercice", "bain", "massage", "démonstration")):
        return "Vidéo"
    if any(w in msg for w in ("sommeil", "témoignage", "histoire", "parcours")):
        return "Podcast"
    if any(w in msg for w in ("nutrition", "alimentation", "droits", "calendrier")):
        return "Article"

    # Défaut : Vidéo (format le plus engageant)
    return "Vidéo"


def _infer_thematique(message: str) -> str:
    """
    Déduit la thématique du message pour affiner la recherche multimédia.
    """
    msg = message.lower()

    thematique_map = {
        "allaitement": ("allaitement", "lait", "sein", "tétée", "lactation",
                        "crevasse", "engorgement"),
        "sommeil bébé": ("sommeil", "nuit", "dodo", "endormissement", "réveil"),
        "yoga prénatal": ("yoga", "exercice prénatal", "stretching"),
        "bain bébé": ("bain", "baigner"),
        "nutrition prénatale": ("nutrition", "alimentation grossesse", "manger enceinte",
                                "nausée", "fringale"),
        "post-partum": ("post-partum", "après accouchement", "baby blues",
                        "dépression", "rééducation", "périnée"),
        "portage": ("portage", "écharpe", "porte-bébé"),
        "massage bébé": ("massage", "colique", "bien-être bébé"),
        "nutrition bébé": ("diversification", "alimentation bébé", "purée"),
    }

    for thematique, keywords in thematique_map.items():
        if any(kw in msg for kw in keywords):
            return thematique

    return ""


# ──────────────────────────────────────────────
# 4. Application FastAPI
# ──────────────────────────────────────────────

bot: PerinatalBot | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialise le PerinatalBot une seule fois au démarrage du serveur."""
    global bot
    bot = PerinatalBot()
    print("✅ Ti'Konsey — PerinatalBot initialisé (v3.0 multimédia)")
    yield
    bot = None


app = FastAPI(
    title="Ti'Konsey API",
    description="API d'accompagnement périnatal pour La Réunion (974) — avec contenus multimédia",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatResponse(BaseModel):
    reponse: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(profile: UserProfile) -> ChatResponse:
    """
    Reçoit un profil utilisateur + message en JSON.
    Retourne la réponse de Béa avec :
    - une recommandation de ressource professionnelle locale
    - un contenu éducatif multimédia complémentaire (Vidéo, Podcast ou Article)
    """
    if bot is None:
        raise HTTPException(status_code=503, detail="Le bot n'est pas encore initialisé.")

    try:
        reponse = await bot.chat(profile)
        return ChatResponse(reponse=reponse)
    except Exception as exc:
        print(f"❌ Erreur Mistral AI : {exc}")
        raise HTTPException(
            status_code=500,
            detail="Une erreur est survenue lors du traitement de votre message. Réessayez dans quelques instants.",
        ) from exc


@app.get("/health")
async def health():
    return {"status": "ok", "bot_ready": bot is not None}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)