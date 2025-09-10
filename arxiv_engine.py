"""Core engine for fetching, scoring, and recommending arXiv papers.

This module defines the ArxivEngine class, which encapsulates all the logic
for learning user interests from a list of favorite papers, fetching new
papers from arXiv RSS feeds, and scoring them based on a combination of
author preference and semantic similarity of the content.
"""
import sys
import time
import pickle
import os
import re
import configparser

import feedparser
from pyzotero import zotero
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

from config import (
    BASE, EMBEDDING_MODEL, CACHE_FILE, CONFIG_FILE, RSS_URLS,
    AUTHOR_WEIGHT, SEMANTIC_WEIGHT
)


__author__ = 'Lijing Shao'
__email__ = 'Friendshao@gmail.com'
__licence__ = 'GPL'

#---------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
#---------------------------------------------------------------------

class ArxivEngine(object):
    """arXiv class for fetching and scoring papers.

    This class handles the main logic of the recommendation system. It can be
    initialized in two modes: 'update' to learn from favorite papers and
    generate interest models, or 'feed' to fetch and score daily papers.

    Attributes:
        mode (str): The operational mode, either 'update' or 'feed'.
        model (SentenceTransformer): The loaded sentence-transformer model for encoding text.
        author_scores (dict): A dictionary mapping author names to their learned scores.
        interest_vector (np.ndarray): A numpy array representing the user's learned interest vector.
    """

    def __init__(self, mode='feed'):
        """Initializes the ArxivEngine.

        Args:
            mode (str, optional): The operational mode. Can be 'update' to rebuild
                the interest models, or 'feed' to prepare for fetching daily papers.
                Defaults to 'feed'.
        """
        self.mode = mode
        self.config = self._load_config()
        self.zotero_client = self._get_zotero_client()
        self.favorite_papers = None # Cache for favorite papers

        self._ensure_nltk_data()

        # Load the model once during initialization to be reused.
        # This is more efficient and avoids rate-limiting errors from HuggingFace.
        print("Loading sentence-transformer model... (this may take a moment on first run)")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded successfully.")

        if self.mode == 'update':
            self.update_interest_model()
        
        self.author_scores, self.interest_vector = self.load_score_files()

    def _load_config(self):
        """Loads configuration from config.ini."""
        config = configparser.ConfigParser()
        if not os.path.exists(CONFIG_FILE):
            print("Warning: config.ini not found. Zotero integration will be disabled.")
            return None
        config.read(CONFIG_FILE)
        return config

    def _get_zotero_client(self):
        """Initializes and returns a Zotero client if config is valid."""
        if not self.config or 'zotero' not in self.config:
            return None
        
        try:
            user_id = self.config.get('zotero', 'user_id')
            api_key = self.config.get('zotero', 'api_key')
            if not user_id or not api_key or 'YOUR' in user_id or 'YOUR' in api_key:
                raise ValueError("Zotero user_id or api_key is not set in config.ini")
            return zotero.Zotero(user_id, 'user', api_key)
        except (configparser.NoOptionError, configparser.NoSectionError, ValueError) as e:
            print(f"Warning: Zotero config is incomplete or invalid in config.ini: {e}")
            return None

    def _ensure_nltk_data(self):
        """Checks for and downloads NLTK 'punkt' data if missing."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("NLTK 'punkt' tokenizer not found. Downloading...")
            nltk.download('punkt', quiet=True)
            print("Download complete.")

    def clean_text(self, text):
        """Cleans a given text string by removing punctuation and extra spaces.

        Args:
            text (str): The input string to clean.

        Returns:
            str: The cleaned text.
        """
        for s in '''$,'.()[]{}?!&*/\`~<>^%-+|"''':
            text = text.replace(s, ' ')
        text = [x.strip() for x in text.split(' ') if len(x.strip()) > 0]
        return ' '.join(text)

    def get_favorite_papers_from_api(self):
        """Fetches and caches favorite papers from the Zotero API."""
        if self.favorite_papers is not None:
            return self.favorite_papers

        if not self.zotero_client:
            print("Zotero client not configured. Cannot fetch favorite papers.")
            self.favorite_papers = []
            return self.favorite_papers

        collection_id = self.config.get('zotero', 'collection_id', fallback=None)
        
        print(f"Fetching items from Zotero library...")
        try:
            if collection_id:
                print(f"Using collection ID: {collection_id}")
                items = self.zotero_client.collection_items(collection_id)
            else:
                print("Fetching all items from the library.")
                items = self.zotero_client.everything(self.zotero_client.items())
        except Exception as e:
            print(f"Error fetching from Zotero API: {e}")
            self.favorite_papers = []
            return self.favorite_papers
        
        print(f"Found {len(items)} items in Zotero.")

        papers = []
        for item in items:
            data = item.get('data', {})
            # We are interested in journal articles, conference papers, preprints, etc.
            if data.get('itemType') not in ['journalArticle', 'conferencePaper', 'preprint', 'report', 'thesis', 'bookSection', 'book']:
                continue

            title = data.get('title', '')
            abstract = data.get('abstractNote', '')
            
            authors_list = []
            creators = data.get('creators', [])
            for creator in creators:
                if creator.get('creatorType') == 'author':
                    first_name = creator.get('firstName', '')
                    last_name = creator.get('lastName', '')
                    authors_list.append(f"{first_name} {last_name}".strip())
            
            if title and authors_list:
                papers.append({
                    'title': title,
                    'abstract': abstract,
                    'authors': authors_list
                })
        
        self.favorite_papers = papers
        return self.favorite_papers

    def update_interest_model(self):
        """Updates the interest model based on a Zotero library.

        This method reads from the Zotero API, processes each paper to
        extract author and content information, and then builds two model files:
        - author.pickle: Contains scores for authors based on their frequency in the favorite papers.
        - interest_vector.pickle: A single vector representing the semantic center of the user's interests.
        """
        # --- Step 1: Check for changes in Zotero library ---
        last_version = -1
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
                last_version = cached_data.get('zotero_version', -1)
        
        try:
            current_version = self.zotero_client.last_modified_version()
            if current_version == last_version:
                print(f"Zotero library has not changed (version {current_version}). Skipping update.")
                return
        except Exception as e:
            print(f"Warning: Could not check Zotero library version: {e}. Proceeding with full update.")
            current_version = -1 # Force update if version check fails

        # --- Step 2: Fetch papers from API ---
        papers = self.get_favorite_papers_from_api()

        if not papers:
            print("No papers loaded from Zotero. Aborting update.")
            return
        author_scores = {}
        texts_to_embed = []

        print(f"Processing {len(papers)} papers from Zotero library...")
        for paper in papers:
            # Update author scores
            num_authors = len(paper['authors'])
            if num_authors > 0:
                for author_name in paper['authors']:
                    author_scores[author_name] = author_scores.get(author_name, 0.0) + 1.0 / num_authors
            
            texts_to_embed.append(paper['title'] + ' ' + paper['abstract'])

        # Create and store embeddings with a progress bar
        print("Generating embeddings for interest model...")
        favorite_embeddings = self.model.encode(
            texts_to_embed, 
            show_progress_bar=True, 
            convert_to_tensor=False
        )

        if len(favorite_embeddings) > 0:
            interest_vector = np.mean(favorite_embeddings, axis=0)
        else:
            interest_vector = None

        # --- Step 3: Save all results to a single cache file ---
        print(f"Saving updated models to '{os.path.basename(CACHE_FILE)}'...")
        cache_data = {
            'zotero_version': current_version,
            'papers': papers,
            'author_scores': author_scores,
            'interest_vector': interest_vector,
            'embeddings': favorite_embeddings
        }
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)

        print("\nInterest model updated successfully.")

        # Print Top20 authors
        if author_scores:
            top20_authors = sorted(author_scores, key=author_scores.get, reverse=True)[:20]
            print('\n\t\t === Top 20 Favorite Authors === \n')
            for author in top20_authors:
                print(f"{author.rjust(40)} : {author_scores[author]:.3f}")

        # Run analysis after updating the model
        print("\n--- Running Analysis of Favorite Papers ---")
        from analyze_favorites import run_analysis as run_favorites_analysis
        run_favorites_analysis(self)

    def load_score_files(self):
        """Loads the author scores and the interest vector from pickle files.

        Returns:
            tuple[dict, np.ndarray | None]: A tuple containing the author scores
            dictionary and the interest vector numpy array. The interest vector
            can be None if the file doesn't exist.
        """
        if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
            with open(CACHE_FILE, 'rb') as f:
                try:
                    cache_data = pickle.load(f)
                    author_scores = cache_data.get('author_scores', {})
                    interest_vector = cache_data.get('interest_vector', None)
                    # Note: We don't load 'papers' or 'embeddings' here as they are not needed for scoring.
                    return author_scores, interest_vector
                except (EOFError, pickle.UnpicklingError) as e:
                    print(f"Warning: Could not load cache file '{os.path.basename(CACHE_FILE)}'. It might be corrupted. Error: {e}")
        
        return {}, None

    def score_from_entry(self, entry=dict(),
                         load_score=True, score_files=(None, None),
                         verbose=False):
        """Calculates a recommendation score for a given feedparser entry.

        The score is a weighted combination of an author score (based on the
        paper's authors appearing in the learned author model) and a semantic
        score (based on the cosine similarity between the paper's content
        embedding and the user's interest vector).

        Args:
            entry (dict, optional): A feedparser entry dictionary. Defaults to dict().
            load_score (bool, optional): If True, loads score files from instance
                attributes. If False, uses `score_files`. Defaults to True.
            score_files (tuple, optional): A tuple containing (author_scores, interest_vector)
                to use if `load_score` is False. Defaults to (None, None).
            verbose (bool, optional): Unused in the current implementation. Defaults to False.

        Returns:
            tuple[float, list[float]]: A tuple containing the total score and a list
            of the component scores [author_score, semantic_score].
        """
        if load_score:
            author_scores, interest_vector = self.author_scores, self.interest_vector
        else:
            author_scores, interest_vector = score_files

        if interest_vector is None:
            print("Interest vector not found. Please run with 'update' mode first.")
            return 0.0, [0.0, 0.0]
        content = self.get_content_from_entry(entry)

        # 1. Calculate Author Score
        author_score = 0.0
        for author_name in content['author']:
            author_score += author_scores.get(author_name, 0.0)

        # 2. Calculate Semantic Score
        text_to_embed = content['title'] + ' ' + content['abstract']
        entry_embedding = self.model.encode(text_to_embed, convert_to_tensor=True)
        semantic_score = util.cos_sim(interest_vector, entry_embedding).item()

        # 3. Combine scores
        total_score = (author_score * AUTHOR_WEIGHT) + (semantic_score * SEMANTIC_WEIGHT)

        # Penalize replacements and cross-lists
        summary_lower = entry['summary'].lower()
        if 'announce type: replace' in summary_lower or 'announce type: cross' in summary_lower:
            total_score *= 0.5

        return total_score, [author_score, semantic_score]

    def get_content_from_entry(self, entry=dict()):
        """Extracts and cleans content from a feedparser entry.

        Args:
            entry (dict, optional): A feedparser entry dictionary. Defaults to dict().

        Returns:
            dict: A dictionary with 'author', 'title', and 'abstract' keys.
                  The author value is a dict of author names and their counts.
        """
        content = {'author' : dict(), 'title' : '', 'abstract' : ''}
        # title
        title = re.sub('\(.*?\)', '', entry['title'])[:-2].replace('\n', ' ')
        content['title'] = self.clean_text(title.lower())
        # abstract
        abstract = re.sub(r'\<[^>]*\>', '', entry['summary']).replace('\n', ' ')
        content['abstract'] = self.clean_text(abstract.lower())
        # author
        author = re.sub(r'\<[^>]*\>', '', entry['author'])  # rm '<...>'
        author = author.split('et al')[0]  # trim
        author = re.sub('\(.*?\)', '', author)  # rm affiliation
        author = author.replace('\n', ' ').split(',')
        author = [x.strip() for x in author]
        for x in author:
            content['author'][x] = content['author'].get(x, 0) + 1
        return content

    def _generate_tldr(self, text):
        """Generates a one-sentence summary (TLDR) for a given text.

        This uses a simple but effective heuristic: find the sentence that is most
        semantically similar to the entire text.
        """
        try:
            # Clean up text and split into sentences
            text = re.sub(r'\s+', ' ', text).strip()
            sentences = sent_tokenize(text)
            # Filter out very short sentences that are unlikely to be meaningful
            sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            if not sentences:
                return text[:250] + "..." if len(text) > 250 else text

            text_embedding = self.model.encode(text, convert_to_tensor=True)
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            similarities = util.cos_sim(text_embedding, sentence_embeddings)
            return sentences[np.argmax(similarities)]
        except Exception as e:
            print(f"  ... TLDR generation failed: {e}")
            return text[:250] + "..." if len(text) > 250 else text

    def get_recommendations(self, max_papers, min_score):
        """Fetches daily papers, scores them, and returns a sorted list of recommendations.

        This method iterates through the RSS_URLS, fetches all new papers,
        scores each one, and then filters and sorts them based on the provided
        thresholds.

        Args:
            max_papers (int): The maximum number of papers to return.
            min_score (float): The minimum score a paper must have to be included.

        Returns:
            list[dict]: A list of recommended paper items, where each item is a
            dictionary containing the score, score components, and the original feedparser entry.
        """
        links, scored_entries = [], []
        # Prepare entries with scores
        print("Fetching and scoring papers from RSS feeds...")
        for rss_url in tqdm(RSS_URLS, desc="Parsing Feeds"):
            feed = feedparser.parse(rss_url)
            for entry in feed["entries"]:
                link = entry['link']
                if link not in links:
                    links.append(link)
                    score, score_list = self.score_from_entry(entry, load_score=False, score_files=(self.author_scores, self.interest_vector))
                    scored_entries.append({
                        'score': score,
                        'score_list': score_list,
                        'entry': entry
                    })

        # Sort entries by score in descending order
        sorted_entries = sorted(scored_entries, key=lambda x: x['score'], reverse=True)

        # First, select the top papers that meet the threshold
        papers_to_process = []
        for item in sorted_entries:
            if len(papers_to_process) >= max_papers:
                break
            if item['score'] >= min_score:
                papers_to_process.append(item)

        papers_to_show = []
        # Now, generate TLDRs only for the selected papers, with a progress bar
        if papers_to_process:
            for item in tqdm(papers_to_process, desc="Generating TLDRs"):
                # Clean the summary text before generating TLDR
                summary_text = re.sub(r'<[^>]*>', '', item['entry']['summary'])
                summary_text = re.sub(r'\s+', ' ', summary_text).strip()
                item['tldr'] = self._generate_tldr(summary_text)
                papers_to_show.append(item)

        print(f'\nIn total: {len(scored_entries)} entries, recommending top {len(papers_to_show)}\n')
        return papers_to_show


if __name__ == '__main__':
    # This block allows the script to be run directly to update the interest model.
    if len(sys.argv) > 1 and sys.argv[1] == 'update':
        print("Updating interest model...")
        ArxivEngine(mode='update')
        print("Update complete.")
    else:
        print("This script contains the core recommendation engine and is intended to be imported.")
        print("To update the model, run: python arxiv_engine.py update")
        print("To generate the daily HTML page, run: python arxiv_rank.py")