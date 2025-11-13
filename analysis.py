"""
A tool for analyzing and visualizing arXiv papers.

This script contains functions for:
- Clustering favorite papers from a Zotero library.
- Generating visualizations like cluster maps and word clouds.
- Plotting score distributions for daily papers.

Usage:
    python analysis.py [number_of_clusters]
Example:
    python analysis.py 7
"""
import os
import sys
import pickle
import textwrap
import warnings
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

# Add the project root to the path to allow importing arxiv_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE, CACHE_FILE, ANALYSIS_OUTPUT_DIR, CLUSTER_NAMING_METHOD, WORD_CLOUD_METHOD, LLM_PROVIDER,
    LLM_MODEL
)
from llm_utils import query_llm
from arxiv_paper import ArxivPaper

# --- Constants ---
# Define custom stopwords to be used by both TF-IDF and WordCloud for consistency.
CUSTOM_STOPWORDS = {
    "paper", "study", "results", "show", "model", "based", "using", "propose",
    "method", "approach", "data", "analysis", "figure", "table", "et", "al",
    "fig", "also", "however", "therefore", "provide", "present", "shown",
    "within", "different", "can", "due", "may", "well", "new", "large", "high"
}

MIN_PAPERS_FOR_SUBCLUSTERING = 30


def load_and_process_papers():
    """
    Loads favorite papers from the local cache and generates embeddings.
    The cache is created by running `python arxiv_engine.py update`.

    Returns:
        tuple: A tuple containing:
            - papers (list[dict]): A list of paper dictionaries.
            - embeddings (np.ndarray): An array of embeddings for the papers.
            - zotero_version (int): The version of the Zotero library from the cache.
    """
    print("\n--- Loading Favorite Papers and Embeddings from Unified Cache ---")
    if not os.path.exists(CACHE_FILE):
        print(f"Error: Cache file '{os.path.basename(CACHE_FILE)}' not found.")
        print("Please run 'python arxiv_engine.py update' first to fetch data from Zotero and create the cache.")
        sys.exit(1)

    with open(CACHE_FILE, 'rb') as f:
        cached_data = pickle.load(f)
        papers = cached_data.get('papers', [])
        embeddings = cached_data.get('embeddings', None)
        zotero_version = cached_data.get('zotero_version', -1)

    if not papers or embeddings is None:
        print("No papers found in the local cache. The cache might be empty or corrupted.")
        print("Please run 'python arxiv_engine.py update' to regenerate it.")
        sys.exit(1)

    print(f"\n--- Found and loaded {len(papers)} favorite papers and their embeddings from cache. ---")

    return papers, embeddings, zotero_version

def get_cluster_names(papers, labels, n_clusters):
    """
    Generates meaningful names for each cluster. Dispatches to the appropriate method.

    Args:
        papers (list[dict]): A list of paper dictionaries.
        labels (np.ndarray): The cluster label for each paper.
        n_clusters (int): The total number of clusters.

    Returns:
        list[str]: A list of generated names for each cluster.
    """
    if CLUSTER_NAMING_METHOD == 'llm' and n_clusters > 1:
        names = get_cluster_names_llm(papers, labels, n_clusters)
        # Check if LLM actually generated meaningful names (not just default "Cluster X")
        if names and not all(name.startswith('Cluster ') for name in names):
            return names
        print("Warning: LLM cluster naming failed or returned only default names. Falling back to TF-IDF method.")

    return get_cluster_names_tfidf(papers, labels, n_clusters)

def get_cluster_names_llm(papers, labels, n_clusters):
    """Generates cluster names using an LLM."""
    print("Generating meaningful cluster names using LLM...")
    cluster_texts = ['' for _ in range(n_clusters)]
    for i, paper in enumerate(papers):
        cluster_texts[labels[i]] += f"Title: {paper['title']}\nAbstract: {paper['abstract'][:500]}...\n\n"

    cluster_names = []
    failed_count = 0
    for i in range(n_clusters):
        if not cluster_texts[i]:
            cluster_names.append(f"Cluster {i} (empty)")
            continue
        
        text_for_prompt = cluster_texts[i][:8000]
        prompt = (
            "You are an expert research scientist. Based on the following paper titles and abstracts, "
            "generate a short, descriptive name for this research cluster. The name should be 2-4 keywords "
            "that capture the main theme. Return the name as a single string, with keywords separated by commas. "
            "Example: 'dark matter, simulation, halos'.\n\n"
            f"--- PAPERS ---\n{text_for_prompt}"
        )
        name = query_llm(prompt, model_name=LLM_MODEL, temperature=0.1, max_tokens=20)
        if name:
            cluster_names.append(name.strip().replace('"', ''))
        else:
            cluster_names.append(f"Cluster {i}")
            failed_count += 1
    
    print("Generated names (LLM):", cluster_names)
    if failed_count > 0:
        print(f"Warning: {failed_count}/{n_clusters} cluster names failed to generate from LLM.")
    return cluster_names

def get_cluster_names_tfidf(papers, labels, n_clusters):
    """
    Generates meaningful names for each cluster based on TF-IDF of paper content.

    Args:
        papers (list[dict]): A list of paper dictionaries.
        labels (np.ndarray): The cluster label for each paper.
        n_clusters (int): The total number of clusters.

    Returns:
        list[str]: A list of generated names for each cluster.
    """
    print("Generating meaningful cluster names...")
    cluster_texts = ['' for _ in range(n_clusters)]
    for i, paper in enumerate(papers):
        cluster_texts[labels[i]] += paper['title'] + ' ' + paper['abstract'] + ' '

    # Use scikit-learn's built-in English stop words and add our custom ones.
    # This is more robust and avoids the UserWarning.
    stopwords = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)

    vectorizer = TfidfVectorizer(max_features=1000, stop_words=list(stopwords), ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
    feature_names = vectorizer.get_feature_names_out()

    cluster_names = []
    for i in range(n_clusters):
        # Get the i-th row (which is a sparse matrix itself) and convert only it to a dense array.
        # This is much more memory-efficient than calling .toarray() on the whole matrix.
        row_dense = tfidf_matrix.getrow(i).toarray().flatten()
        # Get indices of the top 3 terms by score
        top_term_indices = row_dense.argsort()[-3:][::-1]
        # Filter out terms with a score of 0 and create the name
        top_terms = [feature_names[j] for j in top_term_indices if row_dense[j] > 0]
        cluster_names.append(', '.join(top_terms) if top_terms else f"Cluster {i}")
            
    print("Generated names:", cluster_names)
    return cluster_names

def plot_clusters(reduced_embeddings, hierarchical_labels, n_clusters, hierarchical_cluster_names, naming_method='tfidf'):
    """
    Generates and saves a 2D scatter plot of the paper clusters, showing sub-clusters.

    Args:
        reduced_embeddings (np.ndarray): 2D coordinates of the papers.
        hierarchical_labels (list[tuple]): Hierarchical cluster label (p_id, s_id) for each paper.
        n_clusters (int): The number of primary clusters.
        hierarchical_cluster_names (dict): A dict mapping (p_id, s_id) to a name.
        naming_method (str): The method used for cluster naming ('tfidf' or 'llm').
    """
    print("Generating hierarchical cluster plot with sub-cluster coloring...")
    plt.figure(figsize=(22, 18))  # Increased size for larger fonts and legend

    labels_np = np.array(hierarchical_labels)
    primary_labels = labels_np[:, 0]
    sub_labels = labels_np[:, 1]

    # Define base colormaps for primary clusters for better visual grouping
    base_cmaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'Greys', 'YlOrBr', 'BuPu', 'GnBu', 'YlGn']
    
    legend_handles = []

    # Plot each sub-cluster group individually
    for p_id in range(n_clusters):
        primary_mask = (primary_labels == p_id)
        if not np.any(primary_mask):
            continue

        # Get colormap for this primary cluster
        cmap = plt.cm.get_cmap(base_cmaps[p_id % len(base_cmaps)])
        
        unique_sub_labels_in_primary = sorted(np.unique(sub_labels[primary_mask]))
        num_sub_clusters = len(unique_sub_labels_in_primary)
        
        # Generate distinct colors for each sub-cluster from the primary's colormap
        sub_colors = cmap(np.linspace(0.4, 0.95, num_sub_clusters))

        for i, s_id in enumerate(unique_sub_labels_in_primary):
            color = sub_colors[i]
            mask = primary_mask & (sub_labels == s_id)
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                color=color,
                s=120,  # Increased point size
                alpha=0.8
            )
            
            # Add handle to legend
            cluster_key = (p_id, s_id)
            cluster_name = hierarchical_cluster_names.get(cluster_key)
            if cluster_name:
                wrapped_name = '\n'.join(textwrap.wrap(cluster_name, width=30))
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', label=wrapped_name,
                               markerfacecolor=color, markersize=15)
                )

    method_label = f"{LLM_PROVIDER}" if naming_method == 'llm' else 'TF-IDF'
    plt.title(f'UMAP Projection of Favorite Papers ({n_clusters} Primary Clusters, Named by {method_label})', fontsize=24)
    plt.xlabel('UMAP Dimension 1', fontsize=18)
    plt.ylabel('UMAP Dimension 2', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if legend_handles:
        # Sort legend handles to group by primary cluster
        legend_handles.sort(key=lambda h: h.get_label())
        plt.legend(handles=legend_handles, loc='best', title="Interest Clusters", fontsize=14, title_fontsize=16)

    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
    
    plot_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'cluster_visualization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Hierarchical cluster plot saved to: {plot_path}")
    return plot_path

def generate_word_cloud(papers):
    """
    Generates and saves a word cloud. Dispatches to the appropriate method.
    """
    if WORD_CLOUD_METHOD == 'llm':
        path = generate_word_cloud_llm(papers)
        if path:  # Fallback if LLM fails
            return path
        print("Warning: LLM word cloud generation failed. Falling back to TF-IDF method.")
    
    return generate_word_cloud_tfidf(papers)

def generate_word_cloud_llm(papers):
    """Generates a word cloud using keywords and weights from an LLM."""
    print("Generating word cloud using LLM...")
    text = ' '.join([p['title'] + ' ' + p['abstract'] for p in papers])
    text_for_prompt = text[:12000]

    prompt = (
        "Analyze the following collection of academic paper abstracts. Extract the 50 most important "
        "and frequent technical keywords and concepts. Return the result as a single JSON object where "
        "keys are the keywords (as strings) and values are their relative importance scores (as integers from 1 to 100).\n\n"
        f"--- ABSTRACTS ---\n{text_for_prompt}"
    )

    frequencies = query_llm(prompt, model_name=LLM_MODEL, temperature=0.1, max_tokens=1000, is_json=True)
    if not frequencies or not isinstance(frequencies, dict):
        print("Error: LLM did not return a valid JSON dictionary for word cloud.")
        return None

    stopwords = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)
    wordcloud = WordCloud(
        width=1600, height=1000, background_color='white',
        stopwords=stopwords, collocations=False, colormap='viridis'
    ).generate_from_frequencies(frequencies)

    plt.figure(figsize=(20, 12)); plt.title(f'Favorite Papers Word Cloud (Keywords by {LLM_PROVIDER})', fontsize=24)
    plt.imshow(wordcloud, interpolation='bilinear'); plt.axis('off')
    
    wordcloud_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'word_cloud.png')
    if not os.path.exists(ANALYSIS_OUTPUT_DIR): os.makedirs(ANALYSIS_OUTPUT_DIR)
    plt.savefig(wordcloud_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Word cloud (Keywords by {LLM_PROVIDER}) saved to: {wordcloud_path}")
    return wordcloud_path

def generate_word_cloud_tfidf(papers):
    """
    Generates and saves a word cloud from the text of all favorite papers.

    Args:
        papers (list[dict]): A list of paper dictionaries.
    """
    print("Generating word cloud...")
    text = ' '.join([p['title'] + ' ' + p['abstract'] for p in papers])
    
    # Use scikit-learn's built-in English stop words and add our custom ones for consistency.
    stopwords = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)

    wordcloud = WordCloud(
        width=1600, height=1000, background_color='white', # Aspect ratio matches figsize
        stopwords=stopwords, collocations=False, colormap='viridis'
    ).generate(text)

    plt.figure(figsize=(20, 12))
    plt.title('Favorite Papers Word Cloud (Keywords by TF-IDF)', fontsize=24)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
        
    wordcloud_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'word_cloud.png')
    plt.savefig(wordcloud_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Word cloud (Keywords by TF-IDF) saved to: {wordcloud_path}")
    return wordcloud_path

def generate_score_distribution_plot(all_scored_papers, min_score_threshold):
    """
    Generates and saves a histogram of the scores of all papers found.

    Args:
        all_scored_papers (list[ArxivPaper]): A list of all scored paper items.
        min_score_threshold (float): The score threshold for recommendations.

    Returns:
        str: The path to the saved plot image, or None if no papers were scored.
    """
    if not all_scored_papers:
        return None

    scores = [paper.score for paper in all_scored_papers]
    
    print("Generating score distribution plot...")
    plt.figure(figsize=(14, 8))
    plt.hist(scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label=f'All {len(scores)} Papers')
    
    plt.axvline(min_score_threshold, color='r', linestyle='--', linewidth=2, 
                label=f'Min Score Threshold: {min_score_threshold:.1f}')
    
    plt.xlabel('Recommendation Score', fontsize=18)
    plt.ylabel('Number of Papers', fontsize=18)
    plt.title('Distribution of Daily Paper Scores', fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.tight_layout()

    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
    
    plot_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'score_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score distribution plot saved to: {plot_path}")
    return plot_path

def run_analysis(papers, embeddings, n_clusters=None):
    """
    Runs the full analysis pipeline and returns paths to the generated plots.

    Args:
        papers (list[dict]): A list of paper dictionaries.
        embeddings (np.ndarray): An array of embeddings for the papers.
        n_clusters (int, optional): The number of clusters to use. If None, it will be
                                    auto-detected. Defaults to None.

    Returns:
        tuple: A tuple containing plot path, wordcloud path, UMAP reducer, 2D embeddings, and cluster labels.
    """


    if not papers:
        return None, None, None, None, None

    # --- Determine the number of clusters ---
    if n_clusters is None:
        print("\n--- Auto-detecting optimal number of clusters... ---")
        best_k = 1
        best_score = -1
        # The number of clusters must be between 2 and n_samples - 1
        k_range = range(2, min(16, len(papers)))
        
        if len(k_range) < 1:
            print("Not enough papers to perform meaningful clustering. Defaulting to 1 cluster.")
            n_clusters = 1
        else:
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                cluster_labels_tmp = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, cluster_labels_tmp)
                print(f"  - Silhouette score for {k} clusters: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k
            n_clusters = best_k
            print(f"Optimal number of clusters found: {n_clusters}")
    
    if len(papers) < n_clusters:
        print(f"Warning: Number of papers ({len(papers)}) is less than the requested number of clusters ({n_clusters}).")
        n_clusters = len(papers)
        if n_clusters == 0:
            print("No papers to analyze.")
            return None, None, None, None, None
        print(f"Adjusting number of clusters to {n_clusters}.")

    print(f"\n--- Performing K-Means clustering with {n_clusters} clusters... ---")
    if n_clusters > 0:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        primary_labels = kmeans.fit_predict(embeddings)
    else:
        primary_labels = np.zeros(len(papers), dtype=int)

    # --- Hierarchical Naming and Sub-clustering ---
    print("\n--- Generating Hierarchical Names and Performing Sub-clustering ---")
    primary_cluster_names_list = get_cluster_names(papers, primary_labels, n_clusters) if n_clusters > 1 else ["All Papers"]
    hierarchical_cluster_names = {}
    for i, name in enumerate(primary_cluster_names_list):
        hierarchical_cluster_names[(i, -1)] = name

    # The final labels will be a list of tuples: (primary_cluster_id, sub_cluster_id)
    # sub_cluster_id is -1 if no sub-clustering is performed.
    hierarchical_labels = [(label, -1) for label in primary_labels]

    for i in range(n_clusters):
        cluster_indices = np.where(primary_labels == i)[0]

        if len(cluster_indices) < MIN_PAPERS_FOR_SUBCLUSTERING:
            print(f"Primary cluster {i} ('{hierarchical_cluster_names.get((i, -1))}') has {len(cluster_indices)} papers. Skipping sub-clustering (threshold: {MIN_PAPERS_FOR_SUBCLUSTERING}).")
            continue

        print(f"Primary cluster {i} ('{hierarchical_cluster_names.get((i, -1))}') has {len(cluster_indices)} papers. Attempting sub-clustering...")
        cluster_embeddings = embeddings[cluster_indices]
        cluster_papers = [papers[j] for j in cluster_indices]

        # Auto-detect optimal number of sub-clusters using silhouette score
        best_k_sub = 1
        best_score_sub = -1
        k_range_sub = range(2, min(10, len(cluster_indices)))  # Limit max sub-clusters for practicality

        if len(k_range_sub) < 1:
            print(f"  - Not enough papers in cluster {i} for meaningful sub-clustering.")
            continue

        for k in k_range_sub:
            kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels_tmp = kmeans_tmp.fit_predict(cluster_embeddings)
            score = silhouette_score(cluster_embeddings, labels_tmp)
            if score > best_score_sub:
                best_score_sub = score
                best_k_sub = k

        if best_k_sub > 1:
            print(f"  - Found {best_k_sub} optimal sub-clusters for primary cluster {i} with score {best_score_sub:.4f}.")
            kmeans_sub = KMeans(n_clusters=best_k_sub, random_state=42, n_init='auto')
            sub_labels = kmeans_sub.fit_predict(cluster_embeddings)

            # Generate and store names for sub-clusters
            sub_cluster_names_list = get_cluster_names(cluster_papers, sub_labels, best_k_sub)
            for s_id, s_name in enumerate(sub_cluster_names_list):
                primary_name = hierarchical_cluster_names.get((i, -1), f"Cluster {i}")
                hierarchical_cluster_names[(i, s_id)] = f"{primary_name} / {s_name}"

            # Update hierarchical labels for the papers in this primary cluster
            for j, sub_label in enumerate(sub_labels):
                original_paper_index = cluster_indices[j]
                hierarchical_labels[original_paper_index] = (i, sub_label)
        else:
            print(f"  - No meaningful sub-clusters found for primary cluster {i} (best k=1).")

    plot_path = None
    reducer = None
    reduced_embeddings = None
    n_neighbors = min(15, len(papers) - 1)
    if n_neighbors >= 2:
        print("\n--- Reducing dimensionality with UMAP... ---")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="umap")
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            plot_path = plot_clusters(reduced_embeddings, hierarchical_labels, n_clusters, hierarchical_cluster_names, naming_method=CLUSTER_NAMING_METHOD)
    else:
        print("\n--- Not enough papers to perform dimensionality reduction. Skipping cluster plot. ---")

    print("\n--- Generating Word Cloud... ---")
    wordcloud_path = generate_word_cloud(papers)

    # NOTE: The returned labels are now hierarchical: list[(primary_label, sub_label)].
    # The calling function (in arxiv_engine.py) will need to be updated to handle this structure.
    return plot_path, wordcloud_path, reducer, reduced_embeddings, hierarchical_labels, hierarchical_cluster_names

def generate_daily_plot(rec_embeddings, rec_labels, output_filename='daily_cluster_map.png', title='Daily Recommendations on Favorite Papers Map'):
    """
    Generates a cluster plot that overlays new recommendations on the base
    hierarchical cluster map of favorite papers, with improved visual distinctions.

    - Base papers are solid, semi-transparent points.
    - Recommended papers are large, hollow points.
    - Primary clusters have distinct colors.
    - Sub-clusters share a color family with their primary cluster but have unique markers.
    - Recommendations match the color and marker of their assigned cluster.

    Args:
        rec_embeddings (list): List of embeddings for recommended papers.
        rec_labels (list[tuple]): List of hierarchical cluster labels (p_id, s_id) for recommended papers.
        output_filename (str): The filename for the output plot.
        title (str): The title for the plot.
    """
    print(f"\n--- Generating Enhanced Overlay Cluster Plot: {title} ---")
    if not rec_embeddings or rec_labels is None:
        print("Warning: No recommendation embeddings or labels provided. Skipping daily map generation.")
        return None

    if not os.path.exists(CACHE_FILE):
        print("Cache file not found. Cannot generate daily plot.")
        return None

    try:
        with open(CACHE_FILE, 'rb') as f:
            cached_data = pickle.load(f)
        
        reducer = cached_data.get('umap_reducer')
        base_embeddings_2d = cached_data.get('umap_embeddings_2d')
        base_labels = cached_data.get('cluster_labels')
        cluster_names = cached_data.get('cluster_names')
        naming_method = cached_data.get('cluster_naming_method', 'tfidf') # Default to tfidf for old caches
        
        if reducer is None or base_embeddings_2d is None or base_labels is None or cluster_names is None:
            print("Cache is missing necessary UMAP/cluster data. Run 'arxiv_engine.py update' to generate it.")
            return None
    except Exception as e:
        print(f"Error loading UMAP data from cache: {e}")
        return None

    # --- Plotting Setup ---
    plt.figure(figsize=(22, 18))
    base_labels_np = np.array(base_labels)

    # Define markers for sub-clusters
    MARKERS = ['o', 's', 'P', 'X', '*', 'D', 'v', '^', '<', '>']

    # Build color and marker maps from the definitive list of all clusters
    primary_to_sub_keys = {}
    for p_id, s_id in cluster_names.keys():
        if p_id not in primary_to_sub_keys:
            primary_to_sub_keys[p_id] = []
        primary_to_sub_keys[p_id].append(s_id)

    base_cmaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'Greys', 'YlOrBr', 'BuPu', 'GnBu', 'YlGn']
    color_map = {}
    marker_map = {}
    
    sorted_primary_ids = sorted(primary_to_sub_keys.keys())
    for p_id in sorted_primary_ids:
        cmap = plt.cm.get_cmap(base_cmaps[p_id % len(base_cmaps)])
        
        sub_ids = sorted(primary_to_sub_keys[p_id])
        num_sub_clusters = len(sub_ids)
        sub_colors = cmap(np.linspace(0.4, 0.95, num_sub_clusters))
        
        for i, s_id in enumerate(sub_ids):
            cluster_key = (p_id, s_id)
            color_map[cluster_key] = sub_colors[i]
            if s_id == -1:
                marker_map[cluster_key] = 'o'  # Default circle for primary-only
            else:
                # Cycle through markers for sub-clusters, excluding the primary-only case
                sub_cluster_index = [sid for sid in sub_ids if sid != -1].index(s_id)
                marker_map[cluster_key] = MARKERS[sub_cluster_index % len(MARKERS)]

    # --- 1. Plot base favorite papers (background) ---
    # Plot each cluster separately to assign markers correctly
    for (p_id, s_id), name in cluster_names.items():
        mask = (base_labels_np[:, 0] == p_id) & (base_labels_np[:, 1] == s_id)
        if not np.any(mask):
            continue
        plt.scatter(
            base_embeddings_2d[mask, 0], base_embeddings_2d[mask, 1],
            c=[color_map.get((p_id, s_id), 'lightgrey')],
            marker=marker_map.get((p_id, s_id), 'o'),
            s=120, 
            alpha=0.5,  # Increased alpha as requested
            label=name
        )

    # --- 2. Project and plot new recommendations (foreground) ---
    print("Projecting new recommendations into the UMAP space...")
    rec_embeddings_2d = reducer.transform(np.array(rec_embeddings))

    for i, (pos, cluster_id) in enumerate(zip(rec_embeddings_2d, rec_labels), 1):
        color = color_map.get(cluster_id, 'grey') # Fallback to grey, no more red
        marker = marker_map.get(cluster_id, 'o') # Default to circle
        
        # Plot a large, hollow marker for the recommendation
        plt.scatter(
            pos[0], pos[1],
            marker=marker,
            facecolors='none',
            edgecolors=[color],
            s=600,
            linewidth=3,
            zorder=10
        )
        # Annotate with rank
        plt.text(pos[0], pos[1], str(i), color='black',
                 fontsize=12, weight='bold', zorder=12, ha='center', va='center')

    base_title = title
    method_label = f"{LLM_PROVIDER}" if naming_method == 'llm' else 'TF-IDF'
    plt.title(f"{base_title}\n(Base map clusters named by {method_label})", fontsize=24)
    plt.xlabel('UMAP Dimension 1', fontsize=18)
    plt.ylabel('UMAP Dimension 2', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- 3. Create a comprehensive legend ---
    legend_handles = []
    sorted_keys = sorted(cluster_names.keys())
    for key in sorted_keys:
        name = cluster_names[key]
        color = color_map.get(key)
        marker = marker_map.get(key, 'o')
        wrapped_name = '\n'.join(textwrap.wrap(name, width=30))
        if name and color is not None:
             legend_handles.append(
                plt.Line2D([0], [0], marker=marker, color='w', label=wrapped_name,
                           markerfacecolor=color, markersize=15)
            )

    rec_handle = plt.Line2D([0], [0], marker='o', color='w', label="Today's Recommendation (hollow)",
                            markerfacecolor='none', markeredgecolor='grey', markeredgewidth=2, markersize=15)
    legend_handles.append(rec_handle)

    plt.legend(handles=legend_handles, loc='best', title="Interest Clusters", fontsize=14, title_fontsize=16)

    # Save plot
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
    
    plot_path = os.path.join(ANALYSIS_OUTPUT_DIR, output_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overlay cluster plot saved to: {plot_path}")
    return plot_path


def main():
    """Main function to run the analysis pipeline as a standalone script."""
    # Load data and Zotero version directly from cache
    papers, embeddings, zotero_version = load_and_process_papers()
    user_n_clusters = None
    if len(sys.argv) > 1:
        try:
            user_n_clusters = int(sys.argv[1])
            print(f"User specified {user_n_clusters} clusters.")
        except ValueError:
            print(f"Usage: python {sys.argv[0]} [number_of_clusters]")
            print("Proceeding with automatic cluster number detection.")
    
    run_analysis(papers, embeddings, n_clusters=user_n_clusters)

    print("\nAnalysis complete. Results are in the 'analysis_results' directory.")

if __name__ == '__main__':
    main()
