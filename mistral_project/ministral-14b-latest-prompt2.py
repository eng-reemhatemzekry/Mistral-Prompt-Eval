import os
import re
import time
import pandas as pd

from dotenv import load_dotenv
try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import Mistral


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv("MISTRAL_MODEL", "ministral-14b-latest")

DATASET_PATH = "dataset_balanced.csv"
OUTPUT_PATH = "results/mistral_large_3_results.csv"

MAX_RETRIES = 10
REQUEST_DELAY = 3.0   # seconds between questions (raise if you still get 429)

VALID_ROUTES = {
    "customer",
    "faq",
    "hybrid",
    "out_of_scope",
}


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

# Always run relative to this script's folder (so .env and dataset are found)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY was not found. "
        "Create a .env file and add your API key."
    )

client = Mistral(api_key=API_KEY)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert banking intent classification system.

Your task is to classify a user's banking question into EXACTLY ONE
of the following four routes:

1. customer
2. faq
3. hybrid
4. out_of_scope

You must classify the INTENT of the user's question, not simply
match keywords.

------------------------------------------------------------
ROUTE DEFINITIONS
------------------------------------------------------------

CUSTOMER
---------
Use "customer" when the user is asking about their own personal
banking/customer-specific information.

Examples:
- "How much money do I have?"
- "What are my recent transactions?"
- "Show me my balance."
- "What is my account number?"
- "Did I receive my salary?"
- "What was my last transaction?"
- "How much did I spend this month?"

The important characteristic is that the answer requires access to
the authenticated customer's personal/account data.

FAQ
---
Use "faq" when the question is a general banking question that can
be answered from public/static banking information and does NOT
require access to the customer's personal data.

Examples:
- "What documents do I need to open an account?"
- "What are the bank's working hours?"
- "What is the minimum balance?"
- "How can I apply for a credit card?"
- "What types of accounts does the bank offer?"
- "What are the requirements for a personal loan?"

The answer should come from general bank policies, products,
procedures, or frequently asked questions.

HYBRID
------
Use "hybrid" when the question combines BOTH:

1. a general banking/product/policy question, AND
2. a request involving the customer's own personal information.

Examples:
- "Do I qualify for a loan based on my salary?"
- "What is the minimum balance, and do I currently meet it?"
- "What are the requirements for a credit card, and do I qualify?"
- "What are the bank's transfer limits and what is my current limit?"
- "How much can I transfer, and how much have I already transferred?"

The key characteristic is that answering the complete question
requires both general banking knowledge and customer-specific data.

OUT_OF_SCOPE
------------
Use "out_of_scope" when the question is unrelated to the supported
banking assistant capabilities.

Examples:
- "What's the weather today?"
- "Write me a Python program."
- "Who won the football match?"
- "Tell me a joke."
- "What is the capital of France?"
- "Help me with my university assignment."

------------------------------------------------------------
IMPORTANT CLASSIFICATION RULES
------------------------------------------------------------

RULE 1:
Classify based on the complete meaning of the question.

RULE 2:
A question about the user's own account, balance, transactions,
profile, cards, payments, or financial activity is usually
"customer" if no general FAQ component is requested.

RULE 3:
A general banking question is "faq" if no customer-specific
information is required.

RULE 4:
If BOTH general banking information AND personal customer data
are required, classify as "hybrid".

RULE 5:
Do not classify based on a single keyword.

RULE 6:
The presence of words such as "my", "account", "card", or
"transaction" does not automatically determine the route.
Understand the entire intent.

RULE 7:
Questions that are unrelated to banking or unsupported by the
banking assistant are "out_of_scope".

RULE 8:
The dataset may contain English, Arabic, Egyptian Arabizi,
or mixed Arabic-English language. Preserve the same intent
classification regardless of language.

Examples of Arabizi:
- "kam flosy fe el hesab?"
- "3ayz a3raf el balance beta3y"
- "momken a3raf akher transaction?"

These should be classified according to their meaning.

------------------------------------------------------------
DECISION PROCESS
------------------------------------------------------------

Internally determine:

1. What is the user asking for?
2. Does answering require customer-specific information?
3. Does answering require general banking/FAQ information?
4. Is the request unrelated to banking?

Then select exactly one route.

Do not explain your reasoning in the final answer.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Return ONLY ONE of these exact labels:

customer
faq
hybrid
out_of_scope

Do not return:
- explanations
- punctuation
- quotes
- markdown
- multiple labels
- reasoning
- additional text

Your entire response must be exactly one route label.
"""


# ============================================================
# CLASSIFICATION FUNCTION
# ============================================================

def classify_question(question: str) -> str:
    """Call Mistral with retry/backoff on rate limits and temporary errors."""

    for attempt in range(MAX_RETRIES):

        try:
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=20,
            )

            answer = response.choices[0].message.content.strip().lower()
            answer = answer.strip("`'\" .,:\n")

            if answer in VALID_ROUTES:
                return answer

            match = re.search(
                r"\b(customer|faq|hybrid|out_of_scope)\b", answer
            )
            return match.group(1) if match else "INVALID"

        except Exception as e:
            msg = str(e).lower()
            retryable = any(
                k in msg for k in ("429", "rate", "503", "502", "500", "timeout")
            )

            if retryable and attempt < MAX_RETRIES - 1:
                wait = min(120, 5 * (2 ** attempt))
                print(f"  Temporary error ({e}). Waiting {wait}s...")
                time.sleep(wait)
                continue

            raise


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("MISTRAL LARGE 3 - BANKING INTENT CLASSIFICATION")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_PATH}")
    print()

    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")

    print(f"Loaded {len(df)} rows.")
    print()

    # Validate expected columns
    required_columns = {"question", "route"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    results = []
    done_ids = set()

    # Resume: keep rows that already succeeded, redo the failed ones
    if os.path.exists(OUTPUT_PATH):
        prev = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
        prev = prev[prev["predicted_route"].isin(VALID_ROUTES)]
        results = prev.to_dict("records")
        done_ids = set(prev["id"].tolist())
        print(f"Resuming: {len(done_ids)} rows already done.\n")

    os.makedirs("results", exist_ok=True)

    for index, row in df.iterrows():

        if (row["id"] if "id" in df.columns else index) in done_ids:
            continue

        question = str(row["question"])
        true_route = str(row["route"]).strip().lower()

        print("-" * 70)
        print(f"Row {index + 1}/{len(df)}")
        print(f"Question: {question}")
        print(f"True route: {true_route}")

        try:

            start_time = time.time()

            predicted_route = classify_question(question)

            latency = time.time() - start_time

            correct = predicted_route == true_route

            print(f"Predicted: {predicted_route}")
            print(f"Correct: {correct}")
            print(f"Latency: {latency:.2f}s")

            results.append(
                {
                    "id": row["id"] if "id" in df.columns else index,
                    "question": question,
                    "language": row["language"]
                    if "language" in df.columns
                    else "",
                    "true_route": true_route,
                    "predicted_route": predicted_route,
                    "correct": correct,
                    "latency_seconds": round(latency, 4),
                    "model": MODEL_NAME,
                }
            )

        except Exception as e:

            print(f"ERROR: {e}")

            results.append(
                {
                    "id": row["id"] if "id" in df.columns else index,
                    "question": question,
                    "language": row["language"]
                    if "language" in df.columns
                    else "",
                    "true_route": true_route,
                    "predicted_route": "ERROR",
                    "correct": False,
                    "latency_seconds": None,
                    "model": MODEL_NAME,
                }
            )

        # Small pause to reduce the chance of hitting rate limits
        # Save after every row so nothing is lost if it stops
        pd.DataFrame(results).to_csv(
            OUTPUT_PATH, index=False, encoding="utf-8-sig"
        )

        time.sleep(REQUEST_DELAY)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs("results", exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    accuracy = results_df["correct"].mean()

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")
    print(f"Total samples: {len(results_df)}")
    print(f"Correct: {results_df['correct'].sum()}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print()
    print(f"Results saved to:")
    print(OUTPUT_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()