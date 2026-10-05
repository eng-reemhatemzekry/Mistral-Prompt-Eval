import os
from dotenv import load_dotenv
try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import Mistral

# Always run relative to this script's folder (so .env and dataset are found)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY was not found in .env")

client = Mistral(api_key=api_key)

response = client.chat.complete(
    model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: MISTRAL API WORKS"
        }
    ],
)

print(response.choices[0].message.content)