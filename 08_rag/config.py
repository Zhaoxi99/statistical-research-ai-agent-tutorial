"""Shared settings. Set OPENAI_API_KEY in a .env file or environment."""
import os
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
EMBED_MODEL = os.getenv('OPENAI_EMBED_MODEL', 'text-embedding-3-small')
CACHE = ROOT / 'cache'
CACHE.mkdir(exist_ok=True)
