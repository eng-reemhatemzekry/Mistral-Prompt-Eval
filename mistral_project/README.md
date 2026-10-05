# Mistral Banking Intent Classification

## Setup
    pip install -r requirements.txt
    # .env must contain: MISTRAL_API_KEY=your_key   (file name is ".env", with the dot)

## Run (in this order)
    python test_mistral.py               # checks key + import
    python check_mistral_models.py       # lists models your key can use
    python mistral_large_3_experiment.py # -> results/mistral_large_3_results.csv
    python mistral_small_4_experiment.py # -> results/mistral_small_4_*.csv/json (resumes if interrupted)

If the small script fails with "model not found", set MODEL_NAME to an id printed by check_mistral_models.py
(e.g. "mistral-small-latest").

## Model not allowed (403 tier_not_allowed)
Run `python find_working_models.py`, then set `MISTRAL_MODEL=<working model>` in `.env`.
