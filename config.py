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

# --- arXiv API Configuration ---
ARXIV_CATEGORIES = [
    'astro-ph.CO', 
    # 'gr-qc', 'hep-ph', 'physics.comp-ph', 'physics.data-an',
    # 'physics.gen-ph', 'physics.hist-ph', 'physics.soc-ph', 'physics.pop-ph',
    # 'math.HO', 'math.PR', 'math.ST', 
    'stat',
    'cs' # Computer Science (CoRR)
]

# Maximum number of results to fetch from arXiv API in a single batch.
# This should be large enough to cover a full day's publications.
MAX_ARXIV_RESULTS = 2000