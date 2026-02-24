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

# Base simulée de ressources locales (mockup)
_MOCK_RESSOURCES: list[dict[str, Any]] = [
    {
        "nom": "PMI de Saint-Denis — Centre Calebassier",
        "type": "PMI",
        "adresse": "Rue du Général de Gaulle, 97400 Saint-Denis",
        "telephone": "0262 90 XX XX",
        "description": "Consultations prénatales, suivi post-partum, vaccinations.",
    },
    {
        "nom": "PMI de Saint-Pierre — Ravine des Cabris",
        "type": "PMI",
        "adresse": "Chemin Ravine des Cabris, 97410 Saint-Pierre",
        "telephone": "0262 35 XX XX",
        "description": "Suivi grossesse, pesée bébé, consultations sage-femme.",
    },
    {
        "nom": "CHU Nord — Félix Guyon (Urgences Maternité)",
        "type": "Hôpital",
        "adresse": "Allée des Topazes, 97400 Saint-Denis",
        "telephone": "0262 90 50 50",
        "description": "Urgences obstétricales, maternité niveau 3.",
    },
    {
        "nom": "CHU Sud — Terre Rouge (Urgences Maternité)",
        "type": "Hôpital",
        "adresse": "BP 350, 97448 Saint-Pierre Cedex",
        "telephone": "0262 35 90 00",
        "description": "Urgences obstétricales, maternité niveau 3.",
    },
    {
        "nom": "Association Naître et Grandir à La Réunion",
        "type": "Association",
        "adresse": "Saint-Denis, 974",
        "telephone": "0693 XX XX XX",
        "description": "Ateliers portage, allaitement, groupes de parole post-partum.",
    },
    {
        "nom": "Réseau Périnatal de La Réunion (REPERE)",
        "type": "Réseau de santé",
        "adresse": "97400 Saint-Denis",
        "telephone": "0262 XX XX XX",
        "description": "Coordination parcours de soins périnatal sur l'île.",
    },
    {
        "nom": "Sage-femme libérale — Marie-Louise Payet",
        "type": "Sage-femme",
        "adresse": "12 rue des Flamboyants, 97410 Saint-Pierre",
        "telephone": "0692 XX XX XX",
        "description": "Préparation à la naissance, suivi post-natal, rééducation périnéale.",
    },
    {
        "nom": "Cabinet de lactation IBCLC — Nathalie Grondin",
        "type": "Consultante en lactation",
        "adresse": "4 rue des Lataniers, 97400 Saint-Denis",
        "telephone": "0693 XX XX XX",
        "description": "Accompagnement allaitement maternel, freins restrictifs, diversification.",
    },
]


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
                        "Recherche des ressources locales (professionnels de santé, "
                        "PMI, associations, hôpitaux) à La Réunion en fonction "
                        "d'un type de ressource et d'une localisation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "type_ressource": {
                                "type": "string",
                                "description": (
                                    "Type de ressource recherchée : "
                                    "'PMI', 'Sage-femme', 'Hôpital', 'Association', "
                                    "'Consultante en lactation', 'Réseau de santé', 'tous'"
                                ),
                            },
                            "localisation": {
                                "type": "string",
                                "description": (
                                    "Commune ou zone de recherche à La Réunion "
                                    "(ex: 'Saint-Denis', 'Saint-Pierre', '974')"
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
    ) -> list[dict[str, Any]]:
        """
        Interroge la base de ressources locales (mockup).

        En production, cette méthode appellera une vraie BDD / API.

        Parameters
        ----------
        type_ressource : filtre par type (PMI, Sage-femme, Hôpital…) ou 'tous'.
        localisation   : filtre optionnel par commune.

        Returns
        -------
        Liste de ressources correspondantes.
        """
        # Simule une latence réseau / BDD
        await asyncio.sleep(0.1)

        resultats = _MOCK_RESSOURCES

        # Filtre par type
        if type_ressource.lower() != "tous":
            resultats = [
                r for r in resultats
                if type_ressource.lower() in r["type"].lower()
            ]

        # Filtre par localisation
        if localisation:
            loc = localisation.lower()
            resultats = [
                r for r in resultats
                if loc in r["adresse"].lower() or loc in r.get("nom", "").lower()
            ]

        return resultats if resultats else [
            {"info": "Aucune ressource trouvée. Contactez le 0262 90 50 50 (CHU Nord) pour être orientée."}
        ]

    # ── Flux conversationnel principal ────────

    async def chat(self, profile: UserProfile) -> str:
        """
        Gère un tour de conversation complet :
        1. Construit le prompt system contextualisé.
        2. Envoie à Mistral avec les outils.
        3. Si l'IA déclenche un tool_call → exécute puis renvoie le résultat.
        4. Retourne la réponse finale.

        Parameters
        ----------
        profile : Profil + message de l'utilisatrice.

        Returns
        -------
        Réponse textuelle du bot.
        """
        # Contextualisation du system prompt
        system_msg = SYSTEM_PROMPT.format(
            age=profile.age,
            stade=profile.stade,
            localisation=profile.localisation,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": profile.message},
        ]

        # — Premier appel à Mistral (peut déclencher un tool_call) —
        response = await self.client.chat.complete_async(
            model=self.MODEL,
            messages=messages,
            tools=self.get_tools(),
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message

        # Si pas de tool_call → réponse directe
        if not assistant_msg.tool_calls:
            return assistant_msg.content

        # — Traitement des tool_calls (boucle pour gérer N appels) —
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

        # — Second appel : Mistral intègre les résultats de l'outil —
        follow_up = await self.client.chat.complete_async(
            model=self.MODEL,
            messages=messages,
            tools=self.get_tools(),
            tool_choice="none",
        )

        return follow_up.choices[0].message.content


# ──────────────────────────────────────────────
# 4. Application FastAPI
# ──────────────────────────────────────────────

# Instance singleton — initialisée au démarrage via le lifespan
bot: PerinatalBot | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialise le PerinatalBot une seule fois au démarrage du serveur."""
    global bot
    bot = PerinatalBot()
    print("✅ Ti'Konsey — PerinatalBot initialisé")
    yield
    bot = None


app = FastAPI(
    title="Ti'Konsey API",
    description="API d'accompagnement périnatal pour La Réunion (974)",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (ouvert pour le développement mobile Expo) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modèle de réponse ──
class ChatResponse(BaseModel):
    reponse: str


# ── Endpoint principal ──
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(profile: UserProfile) -> ChatResponse:
    """
    Reçoit un profil utilisateur + message en JSON,
    retourne la réponse de Ti'Konsey.
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


# ── Health check ──
@app.get("/health")
async def health():
    return {"status": "ok", "bot_ready": bot is not None}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
