"""Core engine for fetching, scoring, and recommending arXiv papers.

This module defines the ArxivEngine class, which encapsulates all the logic
for learning user interests from a list of favorite papers, fetching new
papers from arXiv RSS feeds, and scoring them based on a combination of
author preference and semantic similarity of the content.
"""
import time
import pickle
import os
from datetime import datetime, timedelta, timezone
import re
import argparse
import feedparser
import arxiv

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import nltk

# New imports for the improved TLDR generation
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from config import (
    EMBEDDING_MODEL, CACHE_FILE, AUTHOR_WEIGHT, SEMANTIC_WEIGHT,
    NLTK_DATA_PATH, ARXIV_CATEGORIES, MAX_ARXIV_RESULTS, TLDR_GENERATOR, LLM_PROVIDER, CLUSTER_NAMING_METHOD,
    LLM_MODEL, USE_RECENCY_WEIGHTING, RECENCY_HALF_LIFE_DAYS,
    USE_POSITIONAL_WEIGHTING, FIRST_AUTHOR_BOOST
)
from zotero_client import ZoteroClient
from llm_utils import query_llm
from arxiv_paper import ArxivPaper

def _initialize_nltk():
    """
    Ensures all required NLTK data is downloaded to a local, project-specific
    directory. This function is called once when the module is imported to
    guarantee that dependencies are met before any other code runs.
    This approach is more robust than relying on system-wide installations
    and avoids environment-specific issues.
    """
    # --- Step 1: Define required resources ---
    # We only need 'punkt' and 'stopwords'.
    required_resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords'
    }

    # --- Step 2: Set up local NLTK data path ---
    # Ensure the local NLTK data directory exists
    if not os.path.exists(NLTK_DATA_PATH):
        os.makedirs(NLTK_DATA_PATH)

    # Add the local path to NLTK's data path list
    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_PATH)

    # --- Step 3: Download required resources ---
    for resource_path, resource_name in required_resources.items():
        try:
            nltk.data.find(resource_path, paths=[NLTK_DATA_PATH])
        except LookupError:
            print(f"NLTK resource '{resource_name}' not found. Downloading to '{NLTK_DATA_PATH}'...")
            nltk.download(resource_name, download_dir=NLTK_DATA_PATH, quiet=True)
            print(f"'{resource_name}' download complete.")

    # --- Step 4: Explicitly handle the 'punkt_tab' ghost error ---
    # This is a non-existent resource that has caused persistent errors.
    # We will try to download it as requested to handle the error gracefully.
    try:
        print("Attempting to resolve the 'punkt_tab' issue by trying to download it...")
        nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH, quiet=True)
        print("Warning: 'punkt_tab' was surprisingly found and downloaded. This is unexpected.")
    except ValueError:
        # This is the expected outcome, as 'punkt_tab' is not a real package.
        # We catch the error and inform the user, then proceed.
        print("Confirmed: 'punkt_tab' is not a valid NLTK resource. The error is handled. Continuing...")

# --- Run dependency setup immediately on import ---
_initialize_nltk()

__author__ = 'Lijing Shao'
__email__ = 'Friendshao@gmail.com'
__licence__ = 'GPL'

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
        self.zotero_client = ZoteroClient()

        # Load the model once during initialization to be reused.
        # This is more efficient and avoids rate-limiting errors from HuggingFace.
        print("Loading sentence-transformer model... (this may take a moment on first run)")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded successfully.")

        if self.mode == 'update':
            self.update_interest_model()
        
        self.author_scores, self.interest_vectors, self.interest_similarity_means, self.interest_similarity_stds, self.cluster_names = self.load_score_files()

        # Convert loaded interest models to tensors on the correct device.
        # This now handles both the old list-based format and the new hierarchical dict format.
        if self.interest_vectors is not None:
            if isinstance(self.interest_vectors, dict):
                # New hierarchical format: dict of { (p_id, s_id): data }
                self.interest_vectors = {k: torch.from_numpy(v).to(self.model.device) for k, v in self.interest_vectors.items()}
                if self.interest_similarity_means:
                    self.interest_similarity_means = {k: torch.tensor(v, device=self.model.device, dtype=torch.float32) for k, v in self.interest_similarity_means.items()}
                if self.interest_similarity_stds:
                    self.interest_similarity_stds = {k: torch.tensor(v, device=self.model.device, dtype=torch.float32) for k, v in self.interest_similarity_stds.items()}
            else: # Assumes old list format
                self.interest_vectors = torch.from_numpy(np.array(self.interest_vectors)).to(self.model.device)
                if self.interest_similarity_means is not None:
                    self.interest_similarity_means = torch.from_numpy(np.array(self.interest_similarity_means)).to(self.model.device)
                if self.interest_similarity_stds is not None:
                    self.interest_similarity_stds = torch.from_numpy(np.array(self.interest_similarity_stds)).to(self.model.device)

    def update_interest_model(self):
        """Updates the interest model based on a Zotero library.

        This method reads from the Zotero API, processes each paper to
        extract author and content information, and then builds two model files:
        - author.pickle: Contains scores for authors based on their frequency in the favorite papers.
        - interest_vector.pickle: A single vector representing the semantic center of the user's interests.
        """
        # --- Step 1: Check for changes in Zotero library or the engine script itself ---
        last_zotero_version = -1
        last_engine_mtime = -1
        engine_filepath = os.path.abspath(__file__)
        current_engine_mtime = os.path.getmtime(engine_filepath)
        
        cached_papers_map = {}
        cached_embeddings_map = {}
        old_papers_list = []
        old_embeddings_list = []

        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                try:
                    cached_data = pickle.load(f)
                    last_zotero_version = cached_data.get('zotero_version', -1)
                    last_engine_mtime = cached_data.get('engine_mtime', -1)
                    old_papers_list = cached_data.get('papers', [])
                    old_embeddings_list = cached_data.get('embeddings', [])
                    if old_papers_list and len(old_papers_list) == len(old_embeddings_list):
                        for paper, embedding in zip(old_papers_list, old_embeddings_list):
                            # Ensure cache has the necessary keys for incremental updates
                            if 'key' in paper and 'version' in paper:
                                cached_papers_map[paper['key']] = paper
                                cached_embeddings_map[paper['key']] = embedding
                except (EOFError, pickle.UnpicklingError) as e:
                    print(f"Warning: Cache file is corrupted, forcing full update. Error: {e}")
        
        current_zotero_version = self.zotero_client.get_last_modified_version()

        zotero_unchanged = (current_zotero_version == last_zotero_version) and (current_zotero_version != -1)
        engine_unchanged = (current_engine_mtime == last_engine_mtime)

        if zotero_unchanged and engine_unchanged:
            print(f"Zotero library (v{current_zotero_version}) and arxiv_engine.py are unchanged. Skipping update.")
            return
        elif zotero_unchanged and not engine_unchanged:
            print("arxiv_engine.py has been modified. Forcing an update of the interest model.")
        elif not zotero_unchanged:
            print(f"Zotero library has changed (v{last_zotero_version} -> v{current_zotero_version}). Updating interest model.")

        # --- Step 2: Fetch papers and identify new/modified ones ---
        all_current_papers = self.zotero_client.get_favorite_papers()
        if not all_current_papers:
            print("No papers found in Zotero library. Aborting update.")
            return

        texts_to_embed = []
        paper_data_for_embedding = []
        for paper in all_current_papers:
            key = paper['key']
            # Re-embed if engine script changed, or if paper is new/updated in Zotero.
            if not engine_unchanged or key not in cached_papers_map or paper['version'] > cached_papers_map[key]['version']:
                title_cleaned = ArxivPaper._clean_text(paper['title'], for_embedding=True)
                abstract_cleaned = ArxivPaper._clean_text(paper['abstract'], for_embedding=True)
                texts_to_embed.append(title_cleaned + self.model.tokenizer.sep_token + abstract_cleaned)
                paper_data_for_embedding.append(paper)

        # --- Step 3: Generate embeddings only for the delta ---
        if texts_to_embed:
            print(f"Found {len(texts_to_embed)} new or modified papers. Generating embeddings...")
            new_embeddings = self.model.encode(
                texts_to_embed, 
                show_progress_bar=True, 
                convert_to_tensor=False
            )
            # Update the embeddings map with the newly generated ones
            for paper, embedding in zip(paper_data_for_embedding, new_embeddings):
                cached_embeddings_map[paper['key']] = embedding
        else:
            print("Zotero library version changed, but no new or modified papers found. Re-analyzing existing data.")

        # --- Step 4: Reconstruct full dataset and calculate author scores ---
        papers = []
        embeddings_list = []
        for paper in all_current_papers:
            key = paper['key']
            if key in cached_embeddings_map:
                papers.append(paper)
                embeddings_list.append(cached_embeddings_map[key])
        
        favorite_embeddings = np.array(embeddings_list)

        author_scores = {}
        print(f"Processing {len(papers)} papers from Zotero library for author scores...")
        
        # --- New Author Scoring Logic ---
        now = datetime.now(timezone.utc)
        decay_rate = np.log(2) / RECENCY_HALF_LIFE_DAYS if RECENCY_HALF_LIFE_DAYS > 0 else 0

        for paper in tqdm(papers, desc="Calculating Author Scores"):
            # --- Recency Weighting ---
            recency_weight = 1.0
            if USE_RECENCY_WEIGHTING and decay_rate > 0:
                date_added_str = paper.get('dateAdded')
                if date_added_str:
                    try:
                        # Zotero format is like '2024-01-15T10:00:00Z'
                        date_added = datetime.fromisoformat(date_added_str.replace('Z', '+00:00'))
                        days_since_added = (now - date_added).days
                        recency_weight = np.exp(-decay_rate * max(0, days_since_added))
                    except (ValueError, TypeError):
                        pass # Ignore if date is malformed

            # --- Positional and Recency Weighting ---
            num_authors = len(paper['authors'])
            if num_authors > 0:
                for i, author_name in enumerate(paper['authors']):
                    positional_weight = 1.0
                    if USE_POSITIONAL_WEIGHTING and i == 0:
                        positional_weight = FIRST_AUTHOR_BOOST
                    
                    score_contribution = (positional_weight * recency_weight) / num_authors
                    author_scores[author_name] = author_scores.get(author_name, 0.0) + score_contribution

        # Run analysis before saving the model to get UMAP data
        print("\n--- Running Analysis of Favorite Papers ---")
        from analysis import run_analysis as run_favorites_analysis
        # Call with the new signature, passing data directly
        plot_path, wordcloud_path, reducer, reduced_embeddings, labels, cluster_names = run_favorites_analysis(papers, favorite_embeddings, n_clusters=None)

        # --- Step 5: Calculate hierarchical interest vectors and dispersion ---
        interest_vectors = {}
        interest_similarity_means = {}
        interest_similarity_stds = {}

        if len(papers) > 0 and labels is not None:
            # `labels` is now a list of tuples: (primary_label, sub_label)
            hierarchical_labels_np = np.array(labels)
            primary_labels = hierarchical_labels_np[:, 0]
            unique_primary_labels = np.unique(primary_labels)

            print(f"\nCalculating hierarchical interest vectors for {len(unique_primary_labels)} primary clusters...")

            for p_id in unique_primary_labels:
                # --- Process Primary Cluster ---
                primary_indices = np.where(primary_labels == p_id)[0]
                
                if len(primary_indices) == 0:
                    continue

                primary_embeddings = favorite_embeddings[primary_indices]
                primary_centroid = np.mean(primary_embeddings, axis=0)
                
                # Store primary cluster centroid (key: (primary_id, -1))
                primary_key = (p_id, -1)
                interest_vectors[primary_key] = primary_centroid

                # Calculate and store dispersion for primary cluster
                if len(primary_indices) > 1:
                    centroid_tensor = torch.from_numpy(primary_centroid).unsqueeze(0)
                    cluster_tensor = torch.from_numpy(primary_embeddings)
                    similarities = util.cos_sim(cluster_tensor, centroid_tensor).numpy().flatten()
                    
                    interest_similarity_means[primary_key] = np.mean(similarities)
                    interest_similarity_stds[primary_key] = max(np.std(similarities), 1e-6)
                else:
                    interest_similarity_means[primary_key] = 1.0
                    interest_similarity_stds[primary_key] = 1e-6

                # --- Process Sub-clusters ---
                sub_labels_in_primary = hierarchical_labels_np[primary_indices, 1]
                unique_sub_labels = np.unique(sub_labels_in_primary)

                # Filter out the -1 label which indicates no sub-cluster
                sub_cluster_ids = [s_id for s_id in unique_sub_labels if s_id != -1]

                if not sub_cluster_ids:
                    continue # No sub-clusters for this primary cluster

                print(f"  - Found {len(sub_cluster_ids)} sub-clusters in primary cluster {p_id}.")
                for s_id in sub_cluster_ids:
                    sub_cluster_mask = (sub_labels_in_primary == s_id)
                    original_indices_for_sub = primary_indices[sub_cluster_mask]

                    if len(original_indices_for_sub) == 0: continue
                    
                    sub_embeddings = favorite_embeddings[original_indices_for_sub]
                    sub_centroid = np.mean(sub_embeddings, axis=0)

                    # Store sub-cluster centroid (key: (primary_id, sub_id))
                    sub_key = (p_id, s_id)
                    interest_vectors[sub_key] = sub_centroid

                    # Calculate and store dispersion for sub-cluster
                    if len(original_indices_for_sub) > 1:
                        centroid_tensor = torch.from_numpy(sub_centroid).unsqueeze(0)
                        cluster_tensor = torch.from_numpy(sub_embeddings)
                        similarities = util.cos_sim(cluster_tensor, centroid_tensor).numpy().flatten()
                        
                        interest_similarity_means[sub_key] = np.mean(similarities)
                        interest_similarity_stds[sub_key] = max(np.std(similarities), 1e-6)
                    else:
                        interest_similarity_means[sub_key] = 1.0
                        interest_similarity_stds[sub_key] = 1e-6

            print(f"Generated {len(interest_vectors)} hierarchical interest vectors in total.")

        elif len(favorite_embeddings) > 0: # Fallback for no clustering
            print("\nNo clusters found, using a single global interest vector.")
            global_centroid = np.mean(favorite_embeddings, axis=0)
            interest_vectors[(0, -1)] = global_centroid
            if len(favorite_embeddings) > 1:
                centroid_tensor = torch.from_numpy(global_centroid).unsqueeze(0)
                cluster_tensor = torch.from_numpy(favorite_embeddings)
                similarities = util.cos_sim(cluster_tensor, centroid_tensor).numpy().flatten()
                interest_similarity_means[(0, -1)] = np.mean(similarities)
                interest_similarity_stds[(0, -1)] = max(np.std(similarities), 1e-6)
            else:
                interest_similarity_means[(0, -1)] = 1.0
                interest_similarity_stds[(0, -1)] = 1e-6

        # --- Step 6: Save all results to a single cache file ---
        print(f"Saving updated models to '{os.path.basename(CACHE_FILE)}'...")
        cache_data = {
            'zotero_version': current_zotero_version,
            'papers': papers,
            'engine_mtime': current_engine_mtime,
            'author_scores': author_scores,
            'interest_vectors': interest_vectors,
            'interest_similarity_means': interest_similarity_means,
            'interest_similarity_stds': interest_similarity_stds,
            'embeddings': favorite_embeddings,
            'umap_reducer': reducer,
            'umap_embeddings_2d': reduced_embeddings,
            'cluster_labels': labels,
            'cluster_names': cluster_names,
            'cluster_naming_method': CLUSTER_NAMING_METHOD
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

    def load_score_files(self):
        """Loads the author scores and the interest vector from pickle files.

        Returns:
            tuple: A tuple containing author_scores (dict), interest_vectors (dict/list),
            interest_similarity_means (dict/list), interest_similarity_stds (dict/list),
            and cluster_names (dict). Values can be None/empty if the cache file doesn't exist.
        """
        if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
            with open(CACHE_FILE, 'rb') as f:
                try:
                    cache_data = pickle.load(f)
                    author_scores = cache_data.get('author_scores', {})
                    interest_vectors = cache_data.get('interest_vectors', None)
                    # For backward compatibility, if 'interest_vectors' is not present,
                    # check for the old 'interest_vector' key.
                    if interest_vectors is None:
                        interest_vector = cache_data.get('interest_vector', None)
                        if interest_vector is not None:
                            print("Found legacy 'interest_vector'. Wrapping it in a list for compatibility.")
                            interest_vectors = [interest_vector]
                    interest_similarity_means = cache_data.get('interest_similarity_means', None)
                    interest_similarity_stds = cache_data.get('interest_similarity_stds', None)
                    cluster_names = cache_data.get('cluster_names', {})
                    return author_scores, interest_vectors, interest_similarity_means, interest_similarity_stds, cluster_names
                except (EOFError, pickle.UnpicklingError) as e:
                    print(f"Warning: Could not load cache file '{os.path.basename(CACHE_FILE)}'. It might be corrupted. Error: {e}")
        
        return {}, None, None, None, {}

    def _score_papers_batch(self, papers_to_score, algorithm='z_score'):
        """
        Calculates recommendation scores for a batch of papers efficiently.

        This method leverages batch processing for embedding generation and similarity
        calculation, which is significantly faster than processing papers one by one,
        especially on a GPU.

        Args:
            papers_to_score (list[ArxivPaper]): A list of paper objects to score.
            algorithm (str): The scoring algorithm ('z_score' or 'max_similarity').

        Returns:
            list[ArxivPaper]: The input list, with score-related attributes populated.
        """
        if not papers_to_score:
            return []

        # --- Step 1: Unpack hierarchical model data and prepare for batch processing ---
        # This handles both the new hierarchical dict format and the old list-based format.
        if isinstance(self.interest_vectors, dict):
            if not self.interest_vectors:
                print("Interest vector dictionary is empty. Please run with 'update' mode first.")
                return []
            
            # Unzip the dictionary into ordered lists/tensors
            cluster_keys = list(self.interest_vectors.keys())
            interest_vectors_tensor = torch.stack(list(self.interest_vectors.values()))
            
            means_tensor = None
            stds_tensor = None
            if self.interest_similarity_means and self.interest_similarity_stds:
                # Create tensors from dict values, ordered by cluster_keys to ensure alignment
                means_list = [self.interest_similarity_means[k] for k in cluster_keys]
                stds_list = [self.interest_similarity_stds[k] for k in cluster_keys]
                # self.interest_similarity_means contains 0-dim tensors, so stack them into a 1-D tensor
                means_tensor = torch.stack(means_list)
                stds_tensor = torch.stack(stds_list)
        else: # Handle legacy list-based format for backward compatibility
            cluster_keys = list(range(len(self.interest_vectors))) if self.interest_vectors is not None else []
            interest_vectors_tensor = self.interest_vectors
            means_tensor = self.interest_similarity_means
            stds_tensor = self.interest_similarity_stds

        if interest_vectors_tensor is None or len(interest_vectors_tensor) == 0:
            print("Interest vector(s) not found. Please run with 'update' mode first.")
            return []

        # --- Step 2: Batch embed all papers ---
        texts_to_embed = [p.title_for_embedding + self.model.tokenizer.sep_token + p.summary_for_embedding for p in papers_to_score]
        all_embeddings = self.model.encode(
            texts_to_embed,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=self.model.device
        )

        # --- Step 3: Batch calculate all semantic similarities ---
        # Resulting shape: [num_papers, num_total_clusters]
        all_similarities = util.cos_sim(all_embeddings, interest_vectors_tensor)

        # --- Step 4: Calculate scores for each paper using batched results ---
        for i, paper in enumerate(tqdm(papers_to_score, desc="Calculating Scores")):
            # a. Calculate Author Score
            raw_author_score = sum(self.author_scores.get(author, 0.0) for author in paper.authors)
            author_score = np.tanh(raw_author_score)

            # b. Determine Semantic Score from pre-calculated similarities
            similarities_for_paper = all_similarities[i]

            best_cluster_index = -1
            semantic_score = 0.0

            if algorithm == 'z_score' and means_tensor is not None and stds_tensor is not None:
                # z_scores are calculated against all clusters (primary and sub) at once
                z_scores = (similarities_for_paper - means_tensor) / stds_tensor
                best_cluster_index = torch.argmax(z_scores).item()
                semantic_score = similarities_for_paper[best_cluster_index].item()
            else:
                # Fallback to max_similarity if z_score is not chosen or data is missing
                if algorithm == 'z_score':
                    print("Warning: Dispersion data not found. Falling back to 'max_similarity'.")
                best_cluster_index = torch.argmax(similarities_for_paper).item()
                semantic_score = similarities_for_paper[best_cluster_index].item()
            
            # Get the actual cluster ID (e.g., (2, 1) or 0) using the index
            cluster_id = cluster_keys[best_cluster_index]

            # tqdm.write(f"DEBUG: Paper '{paper.title_text[:50]}...' -> Score: {semantic_score:.3f}, Assigned Cluster ID: {cluster_id}")

            # c. Combine scores
            total_score = (author_score * AUTHOR_WEIGHT) + (semantic_score * SEMANTIC_WEIGHT)
            if paper.status in ['replace', 'cross-list']:
                total_score *= 0.5

            # d. Assign results to the paper object
            paper.score = total_score
            paper.score_list = [author_score, semantic_score]
            paper.embedding = all_embeddings[i].cpu().numpy()
            paper.cluster_id = cluster_id
            paper.cluster_name = self.cluster_names.get(cluster_id, "N/A")

        return papers_to_score

    def _generate_tldr_sumy(self, text):
        """Generates a TLDR using the sumy library (LSA)."""
        try:
            if not text:
                return "No abstract available."

            # sumy setup
            parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")

            # Get the single best sentence
            summary = summarizer(parser.document, 1)

            if summary:
                return str(summary[0])
            else:
                # Fallback if no summary can be generated
                return text[:250] + "..." if len(text) > 250 else text
        except Exception as e:
            print(f"\n[ERROR] TLDR generation failed for a paper: {e}")
            return text[:250] + "..." if len(text) > 250 else text

    def _generate_tldr_llm_single(self, paper):
        """Generates a TLDR for a single paper using an LLM."""
        prompt = f"Summarize the following academic abstract in a single, concise, and informative sentence. Abstract: \"{paper.cleaned_summary}\""
        try:
            summary = query_llm(prompt, model_name=LLM_MODEL, temperature=0.2, max_tokens=8192)
            if summary:
                return summary
            return None # Indicate failure
        except Exception as e:
            print(f"\n[ERROR] LLM single-paper TLDR generation failed for {paper.arxiv_id}: {e}")
            return None

    def _generate_tldrs_batch_llm(self, papers, batch_size=10):
        """
        Generates TLDRs for a list of papers using an LLM. For cloud-based providers
        (OpenAI, Gemini), it uses batch processing. For local providers (llama_cpp),
        it processes papers individually to be more robust.
        """
        provider_label = "local LLM" if LLM_PROVIDER.lower() == 'llama_cpp' else LLM_PROVIDER
        print(f"Generating TLDRs for {len(papers)} papers using {provider_label}...")

        if LLM_PROVIDER.lower() == 'llama_cpp':
            print("Processing papers individually for local LLM.")
            for paper in tqdm(papers, desc="Generating TLDRs"):
                summary = self._generate_tldr_llm_single(paper)
                if summary:
                    paper.tldr = summary + f" ({provider_label})"
                else:
                    # Fallback to sumy if LLM fails
                    print(f"  [FALLBACK] LLM failed for {paper.arxiv_id}. Using 'sumy'.")
                    paper.tldr = self._generate_tldr_sumy(paper.cleaned_summary) + " (sumy fallback)"
            return

        # Existing batch logic for cloud LLMs
        print(f"Processing in batches of {batch_size}...")
        for i in tqdm(range(0, len(papers), batch_size), desc="Processing TLDR Batches"):
            batch = papers[i:i + batch_size]
            
            prompt_parts = [
                "Your task is to summarize multiple paper abstracts.",
                "Provide the output as a single JSON object where each key is the paper's arXiv ID and the value is its one-sentence summary.",
                "Here are the papers to summarize:"
            ]
            
            for paper in batch:
                prompt_parts.append(f"\n--- PAPER ---")
                prompt_parts.append(f"ID: {paper.arxiv_id}")
                prompt_parts.append(f"Abstract: \"{paper.cleaned_summary}\"")
            
            prompt = "\n".join(prompt_parts)
            
            try:
                summaries = query_llm(prompt, model_name=LLM_MODEL, temperature=0.2, max_tokens=8192, is_json=True)
                if not (summaries and isinstance(summaries, dict)):
                    raise ValueError("LLM did not return a valid JSON dictionary.")
                
                # Batch call was successful, assign TLDRs
                for paper in batch:
                    summary_text = summaries.get(paper.arxiv_id, "TLDR generation failed for this paper.")
                    if "failed" not in summary_text:
                        paper.tldr = summary_text + f" ({provider_label})"
                    else:
                        paper.tldr = summary_text

            except Exception as e:
                print(f"\n[WARNING] LLM batch TLDR generation failed for batch starting at index {i}: {e}. Falling back to individual processing for this batch.")
                # Fallback to individual processing for this batch
                for paper in batch:
                    time.sleep(1) # To avoid hitting rate limits
                    single_summary = self._generate_tldr_llm_single(paper)
                    if single_summary:
                        paper.tldr = single_summary + f" ({provider_label} fallback)"
                    else:
                        # Second fallback to sumy
                        print(f"  [FALLBACK] Using 'sumy' for paper {paper.arxiv_id}.")
                        paper.tldr = self._generate_tldr_sumy(paper.cleaned_summary) + " (sumy fallback)"

    def get_recommendations(self, max_papers, min_score, algorithm='z_score'):
        """Fetches the latest daily papers using the arXiv RSS feed, scores them, and returns a sorted list.

        This method uses the efficient two-step process:
        1. Fetch a list of *new* and *cross-listed* paper IDs from the latest arXiv RSS feed.
        2. Fetch the full details for only those IDs using the arXiv API.

        Args:
            max_papers (int): The maximum number of papers to return.
            min_score (float): The minimum score a paper must have to be included.
            algorithm (str): The scoring algorithm to use ('z_score' or 'max_similarity').

        Returns:
            tuple[list[dict], list[dict]]: A tuple containing the list of recommended papers
            and the list of all scored papers.
        """
        # --- Step 1: Fetch daily paper IDs from arXiv RSS feed ---
        print("Fetching daily paper IDs from arXiv RSS feed...")

        # The RSS feed uses '+' as a separator for OR logic
        # The correct format is to join categories with a '+' and without the 'cat:' prefix.
        # e.g., 'astro-ph.CO+astro-ph.GA'
        query = "+".join(ARXIV_CATEGORIES)
        rss_url = f"https://rss.arxiv.org/atom/{query}"
        
        # Use feedparser directly to fetch the RSS feed, as per the reference implementation.
        feed = feedparser.parse(rss_url)
        
        if feed.bozo:
            exception_info = feed.get('bozo_exception', 'The feed is malformed.')
            print(f"Error: Could not parse arXiv RSS feed. Error: {exception_info}")
            print("Aborting recommendation generation.")
            return [], []
        
        if 'Feed error for query' in feed.feed.title:
            print(f"Error: arXiv returned a feed error for the query '{query}'. Please check ARXIV_CATEGORIES in config.py.")
            print("Aborting recommendation generation.")
            return [], []

        # Filter for 'new' announcements only, to strictly follow the reference implementation.
        paper_ids = []
        for entry in tqdm(feed.entries, desc="Parsing RSS feed"):
            # The 'arxiv_announce_type' is a custom tag in the arXiv RSS feed.
            announce_type = getattr(entry, 'arxiv_announce_type', None)
            if announce_type == 'new':
                # Extract ID from 'http://arxiv.org/abs/2401.12345v1' or 'oai:arXiv.org:2401.12345v1' -> '2401.12345'
                match = re.search(r'(\d+\.\d+)', entry.id)
                if match:
                    paper_ids.append(match.group(1))

        if not paper_ids:
            print("No new papers found in the RSS feed for today's announcement. Exiting.")
            return [], []

        # --- Step 2: Fetch full paper details using the arxiv library in batches ---
        print(f"Found {len(paper_ids)} new papers. Fetching full details...")
        client = arxiv.Client(page_size=100, delay_seconds=10, num_retries=5)
        
        # The arxiv library can handle a large id_list and will paginate internally.
        search = arxiv.Search(id_list=paper_ids, sort_by=arxiv.SortCriterion.SubmittedDate)
        
        # Create ArxivPaper objects from the fetched results
        papers_to_score = [ArxivPaper(result) for result in tqdm(client.results(search), total=len(paper_ids), desc="Fetching details")]

        # --- Step 3: Score the fetched papers ---
        scored_papers = self._score_papers_batch(papers_to_score, algorithm=algorithm)
        
        # --- Step 4: Filter, sort, and generate TLDRs ---
        sorted_papers = sorted(scored_papers, key=lambda p: p.score, reverse=True)

        papers_to_process = []
        for paper in sorted_papers:
            if len(papers_to_process) >= max_papers:
                break
            if paper.score >= min_score:
                papers_to_process.append(paper)

        papers_to_show = []
        if papers_to_process:
            print("\n--- Generating TLDRs for top papers ---")
            if TLDR_GENERATOR == 'llm':
                self._generate_tldrs_batch_llm(papers_to_process)
                papers_to_show = papers_to_process
            else: # 'sumy' or other classic methods
                for paper in tqdm(papers_to_process, desc="Generating TLDRs"):
                    paper.tldr = self._generate_tldr_sumy(paper.cleaned_summary) + " (sumy)"
                    papers_to_show.append(paper)

        print(f'\nIn total: {len(scored_papers)} entries, recommending top {len(papers_to_show)}\n')
        return papers_to_show, sorted_papers

    def get_historical_recommendations(self, start_date, end_date, max_papers, min_score, algorithm='z_score'):
        """
        Fetches papers from a specified date range, scores them, and returns a sorted list.

        Args:
            start_date (datetime): The start date of the period to fetch.
            end_date (datetime): The end date of the period to fetch.
            max_papers (int): The maximum number of papers to return.
            min_score (float): The minimum score a paper must have to be included.
            algorithm (str): The scoring algorithm to use ('z_score' or 'max_similarity').

        Returns:
            tuple[list[ArxivPaper], list[ArxivPaper]]: A tuple containing the list of recommended papers
            and the list of all scored papers.
        """
        if self.interest_vectors is None or len(self.interest_vectors) == 0:
            print("Interest vector(s) not found. Please run with 'update' mode first.")
            return [], []
        
        print(f"Fetching papers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        print("This may take a very long time.")

        query_parts = [f"cat:{cat}" for cat in ARXIV_CATEGORIES]
        categories_query = " OR ".join(query_parts)
        
        all_papers_to_score = []
        
        # --- Set up for progress bar over the entire date range ---
        total_days = (end_date - start_date).days
        if total_days < 0: total_days = 0
        
        days_per_chunk = 1  # Process in more efficient monthly chunks
        current_start = start_date
        
        with tqdm(total=total_days + 1, unit="day", desc="Querying historical papers") as pbar:
            while current_start <= end_date:
                chunk_start_str = current_start.strftime('%Y-%m-%d')
                # Define the end of the current chunk
                current_end = current_start + timedelta(days=days_per_chunk - 1)
                if current_end > end_date:
                    current_end = end_date
                chunk_end_str = current_end.strftime('%Y-%m-%d')

                date_query = f"submittedDate:[{current_start.strftime('%Y%m%d')}0000 TO {current_end.strftime('%Y%m%d')}2359]"
                full_query = f"({categories_query}) AND {date_query}"
                
                try:
                    search = arxiv.Search(
                        query=full_query,
                        max_results=float('inf'),
                        sort_by=arxiv.SortCriterion.SubmittedDate
                    )
                    client = arxiv.Client(page_size=200, delay_seconds=5, num_retries=5)
                    results_iterator = client.results(search)
                    
                    # Inner tqdm shows progress for fetching details within the current chunk
                    desc = f"Fetching {chunk_start_str} to {chunk_end_str}"
                    chunk_papers = [ArxivPaper(result) for result in tqdm(results_iterator, desc=desc, leave=False)]
                    
                    all_papers_to_score.extend(chunk_papers)

                except Exception as e:
                    print(f"\nAn error occurred during query for period {chunk_start_str}: {e}")
                
                # --- Update progress bar and set the start for the next chunk ---
                days_processed = (current_end - current_start).days + 1
                pbar.update(days_processed)
                current_start = current_end + timedelta(days=1)

        print(f"\nTotal papers to score from backfill: {len(all_papers_to_score)}")

        # --- Score the fetched papers ---
        scored_papers = self._score_papers_batch(all_papers_to_score, algorithm=algorithm)
        # --- Filter, sort, and generate TLDRs ---
        sorted_papers = sorted(scored_papers, key=lambda p: p.score, reverse=True)

        papers_to_process = []
        for paper in sorted_papers:
            if len(papers_to_process) >= max_papers:
                break
            if paper.score >= min_score:
                papers_to_process.append(paper)

        papers_to_show = []
        if papers_to_process:
            print(f"\n--- Generating TLDRs for top {len(papers_to_process)} recommended papers ---")
            if TLDR_GENERATOR == 'llm':
                self._generate_tldrs_batch_llm(papers_to_process)
                papers_to_show = papers_to_process
            else:
                for paper in tqdm(papers_to_process, desc="Generating TLDRs"):
                    paper.tldr = self._generate_tldr_sumy(paper.cleaned_summary) + " (sumy)"
                    papers_to_show.append(paper)

        print(f'\nIn total: {len(scored_papers)} entries, recommending top {len(papers_to_show)} papers\n')
        return papers_to_show, scored_papers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Core engine for arxiv_daily. Use 'update' to rebuild the interest model from your Zotero library."
    )
    parser.add_argument(
        'command',
        nargs='?',
        choices=['update'],
        help="The command to execute. Currently, only 'update' is supported."
    )
    args = parser.parse_args()

    if args.command == 'update':
        print("Updating interest model...")
        ArxivEngine(mode='update')
        print("Update complete.")
    else:
        print("This script contains the core recommendation engine and is intended to be imported.")
        parser.print_help()
