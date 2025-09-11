"""
Central configuration file for the arXiv Daily project.
"""
import os

# --- Base Paths and Files ---
BASE = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL = 'allenai-specter'
CACHE_FILE = os.path.join(BASE, 'arxiv_cache.pickle')
CONFIG_FILE = os.path.join(BASE, 'config.ini')
NLTK_DATA_PATH = os.path.join(BASE, 'nltk_data')

# --- Analysis Configuration ---
ANALYSIS_OUTPUT_DIR = os.path.join(BASE, 'analysis_results')

# --- Recommendation Engine Configuration ---
AUTHOR_WEIGHT = 5.0
SEMANTIC_WEIGHT = 10.0

# --- Page Generation Configuration ---
MAX_PAPERS_TO_SHOW = 50
MIN_SCORE_THRESHOLD = 3.0
AUTHOR_COLLAPSE_THRESHOLD = 10

# --- arXiv RSS Feeds ---
RSS_URLS = [
    "http://export.arxiv.org/rss/astro-ph",
    "http://export.arxiv.org/rss/gr-qc",
    "http://export.arxiv.org/rss/hep-ph",
    "http://export.arxiv.org/rss/physics.comp-ph",
    "http://export.arxiv.org/rss/physics.data-an",
    "http://export.arxiv.org/rss/physics.gen-ph",
    "http://export.arxiv.org/rss/physics.hist-ph",
    "http://export.arxiv.org/rss/physics.soc-ph",
    "http://export.arxiv.org/rss/physics.pop-ph",
    "http://export.arxiv.org/rss/math.HO",
    "http://export.arxiv.org/rss/math.PR",
    "http://export.arxiv.org/rss/math.ST",
    "http://export.arxiv.org/rss/stat",
    "http://export.arxiv.org/rss/CoRR"
]