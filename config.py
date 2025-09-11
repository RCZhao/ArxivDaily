"""
Central configuration file for the arXiv Daily project.
"""
import os
import configparser

# --- Base Paths and Files ---
BASE = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL = 'allenai-specter'
CACHE_FILE = os.path.join(BASE, 'arxiv_cache.pickle')
CONFIG_FILE = os.path.join(BASE, 'config.ini')
NLTK_DATA_PATH = os.path.join(BASE, 'nltk_data')

# --- Analysis Configuration ---
ANALYSIS_OUTPUT_DIR = os.path.join(BASE, 'analysis_results')

# --- Load settings from config.ini ---
config = configparser.ConfigParser()
# Provide default values for all configurable settings. This makes the script
# runnable even without a config.ini file.
default_config = {
    'settings': {
        'author_weight': '5.0',
        'semantic_weight': '10.0',
        'max_papers_to_show': '50',
        'min_score_threshold': '3.0',
        'author_collapse_threshold': '10',
        'max_arxiv_results': '2000',
        'arxiv_categories': '''
            astro-ph.CO
            astro-ph.GA
            astro-ph.EP
            gr-qc
            hep-ph
            physics.comp-ph
            physics.data-an
            physics.gen-ph
            math.HO
            math.PR
            math.ST
            stat.ML
            cs.AI
        '''
    }
    ,
    'llm': {
        'provider': 'gemini', # or 'openai'
        'api_key': 'YOUR_API_KEY',
        'model': 'gemini-1.5-flash-latest' # or a model like 'gpt-4o-mini'
    },
    'features': {
        'tldr_generator': 'sumy',
        'cluster_naming_method': 'tfidf',
        'word_cloud_method': 'tfidf'
    }
}
config.read_dict(default_config)

if os.path.exists(CONFIG_FILE):
    config.read(CONFIG_FILE)
else:
    print(f"Warning: '{os.path.basename(CONFIG_FILE)}' not found. Using default settings.")

AUTHOR_WEIGHT = config.getfloat('settings', 'author_weight')
SEMANTIC_WEIGHT = config.getfloat('settings', 'semantic_weight')
MAX_PAPERS_TO_SHOW = config.getint('settings', 'max_papers_to_show')
MIN_SCORE_THRESHOLD = config.getfloat('settings', 'min_score_threshold')
AUTHOR_COLLAPSE_THRESHOLD = config.getint('settings', 'author_collapse_threshold')
MAX_ARXIV_RESULTS = config.getint('settings', 'max_arxiv_results')
arxiv_categories_str = config.get('settings', 'arxiv_categories')
ARXIV_CATEGORIES = [line.strip() for line in arxiv_categories_str.strip().split('\n') if line.strip()]

# --- LLM and Feature Flags ---
LLM_PROVIDER = config.get('llm', 'provider')
LLM_API_KEY = config.get('llm', 'api_key')
LLM_MODEL = config.get('llm', 'model')

TLDR_GENERATOR = config.get('features', 'tldr_generator')
CLUSTER_NAMING_METHOD = config.get('features', 'cluster_naming_method')
WORD_CLOUD_METHOD = config.get('features', 'word_cloud_method')