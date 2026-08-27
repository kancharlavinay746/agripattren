import os
import json
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# LOAD ENV
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ENV_FILE = os.path.join(
    PROJECT_ROOT,
    ".env"
)

load_dotenv(ENV_FILE)


# ============================================================
# GROQ CONFIG
# ============================================================

API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
).strip()


# ============================================================
# CLIENT
# ============================================================

def get_client():

    if not API_KEY:
        return None

    try:
        return Groq(api_key=API_KEY)

    except Exception:
        return None


# ============================================================
# STATUS
# ============================================================

def groq_status():

    return {
        "env_file": ENV_FILE,
        "api_key_found": bool(API_KEY),
        "api_key_length": len(API_KEY),
        "model": MODEL,
        "client_available": get_client() is not None
    }


# ============================================================
# TEST GROQ
# ============================================================

def test_groq():

    if not API_KEY:
        return False, "GROQ_API_KEY not found"

    client = get_client()

    if client is None:
        return False, "Groq client could not be created"

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Say hello in one short sentence."
                }
            ],
            temperature=0,
            max_tokens=100
        )

        # ----------------------------------------------------
        # PRINT COMPLETE RESPONSE
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("RAW GROQ RESPONSE")
        print("=" * 70)

        print(response)

        print("\n" + "=" * 70)
        print("RESPONSE TYPE")
        print("=" * 70)

        print(type(response))

        print("\n" + "=" * 70)
        print("CHOICES")
        print("=" * 70)

        print(response.choices)

        if not response.choices:
            return False, "Groq returned no choices"

        choice = response.choices[0]

        print("\n" + "=" * 70)
        print("FIRST CHOICE")
        print("=" * 70)

        print(choice)

        message = choice.message

        print("\n" + "=" * 70)
        print("MESSAGE")
        print("=" * 70)

        print(message)

        print("\n" + "=" * 70)
        print("MESSAGE DICT")
        print("=" * 70)

        try:
            print(message.model_dump())
        except Exception as e:
            print("Could not model_dump:", e)

        content = getattr(
            message,
            "content",
            None
        )

        print("\n" + "=" * 70)
        print("CONTENT")
        print("=" * 70)

        print(repr(content))

        # ----------------------------------------------------
        # CONTENT EXISTS
        # ----------------------------------------------------

        if content:

            return True, str(content).strip()

        # ----------------------------------------------------
        # CHECK REASONING
        # ----------------------------------------------------

        reasoning = getattr(
            message,
            "reasoning",
            None
        )

        print("\n" + "=" * 70)
        print("REASONING")
        print("=" * 70)

        print(repr(reasoning))

        if reasoning:

            return True, str(reasoning).strip()

        return False, "Groq returned empty text"

    except Exception as e:

        print("\n" + "=" * 70)
        print("GROQ EXCEPTION")
        print("=" * 70)

        print(type(e).__name__)
        print(str(e))

        return False, (
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(data):

    rows = data.get("rows", 0)

    columns = data.get(
        "columns",
        []
    )

    relationships = data.get(
        "strong_relationships",
        []
    )

    anomalies = data.get(
        "anomalies",
        0
    )

    prediction = data.get(
        "prediction",
        None
    )

    trends = data.get(
        "trends",
        []
    )

    clusters = data.get(
        "clusters",
        None
    )

    return f"""
You are an agricultural data analyst.

Analyze the following agricultural dataset information.

DATASET

Rows:
{rows}

Columns:
{json.dumps(columns, default=str)}

Strong relationships:
{json.dumps(relationships, default=str)}

Anomalies:
{anomalies}

Trends:
{json.dumps(trends, default=str)}

Clusters:
{json.dumps(clusters, default=str)}

Prediction:
{json.dumps(prediction, default=str)}

Provide a clear report for farmers.

Use these sections:

## 🌾 Overall Agricultural Insight

## 🔗 Relationship Analysis

## 🚨 Anomaly Analysis

## 📈 Trend Analysis

## 🧩 Cluster Analysis

## 🔮 Prediction Insights

## 👨‍🌾 Recommendations for Farmers

## ⚠️ Important Limitations

Do not invent information.
Do not claim correlation means causation.
Use simple language.
"""


# ============================================================
# AI EXPLANATION
# ============================================================

def explain_patterns(data):

    if not API_KEY:

        return "❌ GROQ_API_KEY is missing."

    client = get_client()

    if client is None:

        return "❌ Groq client could not be created."

    prompt = build_prompt(data)

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert agricultural "
                        "data scientist."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )

        if not response:
            return "❌ Groq returned no response."

        if not response.choices:
            return "❌ Groq returned no choices."

        message = response.choices[0].message

        content = getattr(
            message,
            "content",
            None
        )

        if content:

            return str(content).strip()

        # Some reasoning models may put output elsewhere.
        reasoning = getattr(
            message,
            "reasoning",
            None
        )

        if reasoning:

            return str(reasoning).strip()

        return (
            "❌ Groq connected successfully, "
            "but the model returned empty content."
        )

    except Exception as e:

        return (
            f"❌ Groq Error\n\n"
            f"**{type(e).__name__}**\n\n"
            f"`{str(e)}`"
        )


# ============================================================
# ASK AGRICULTURAL QUESTION
# ============================================================

def ask_agri_question(
    question,
    dataset_context=None
):

    if not API_KEY:
        return "❌ GROQ_API_KEY is missing."

    client = get_client()

    if client is None:
        return "❌ Groq client could not be created."

    context = ""

    if dataset_context:

        context = (
            "\nDataset context:\n"
            + json.dumps(
                dataset_context,
                indent=2,
                default=str
            )
        )

    prompt = f"""
You are AgriPattern AI, an agricultural
data analysis assistant.

Answer this question clearly:

{question}

{context}

Do not invent information.
Give practical advice where appropriate.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful agricultural "
                        "AI assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )

        if not response:
            return "❌ Empty response."

        if not response.choices:
            return "❌ No response choices."

        message = response.choices[0].message

        content = getattr(
            message,
            "content",
            None
        )

        if content:
            return str(content).strip()

        reasoning = getattr(
            message,
            "reasoning",
            None
        )

        if reasoning:
            return str(reasoning).strip()

        return "❌ Groq returned empty content."

    except Exception as e:

        return (
            f"❌ Groq Error: "
            f"{type(e).__name__}: {str(e)}"
        )