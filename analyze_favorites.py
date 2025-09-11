"""
A tool for analyzing and visualizing your favorite arXiv papers.

This script performs a clustering analysis on the papers in your Zotero
library. It generates two main outputs in the
'analysis_results' directory:
1. A 2D visualization of the paper clusters (cluster_visualization.png).
2. A word cloud of the most frequent terms (word_cloud.png).

Usage:
    python analyze_favorites.py [number_of_clusters]
Example:
    python analyze_favorites.py 7
"""
import os
import sys
import pickle
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
from config import BASE, CACHE_FILE, ANALYSIS_OUTPUT_DIR

# --- Constants ---
# Define custom stopwords to be used by both TF-IDF and WordCloud for consistency.
CUSTOM_STOPWORDS = {
    "paper", "study", "results", "show", "model", "based", "using", "propose",
    "method", "approach", "data", "analysis", "figure", "table", "et", "al",
    "fig", "also", "however", "therefore", "provide", "present", "shown",
    "within", "different", "can", "due", "may", "well", "new", "large", "high"
}


def load_and_process_papers():
    """
    Loads favorite papers from the local cache and generates embeddings.
    The cache is created by running `python arxiv_engine.py update`.

    Returns:
        tuple: A tuple containing:
            - list[dict]: A list of paper dictionaries with title, abstract, etc.
            - np.ndarray: An array of embeddings for the papers.
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

    if not papers or embeddings is None:
        print("No papers found in the local cache. The cache might be empty or corrupted.")
        print("Please run 'python arxiv_engine.py update' to regenerate it.")
        sys.exit(0)

    print(f"\n--- Found and loaded {len(papers)} favorite papers and their embeddings from cache. ---")

    return papers, embeddings

def get_cluster_names(papers, labels, n_clusters):
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
        row = tfidf_matrix.toarray()[i]
        top_term_indices = row.argsort()[-3:][::-1]
        top_terms = [feature_names[j] for j in top_term_indices if row[j] > 0]
        cluster_names.append(', '.join(top_terms) if top_terms else f"Cluster {i}")
            
    print("Generated names:", cluster_names)
    return cluster_names

def plot_clusters(reduced_embeddings, labels, n_clusters, cluster_names):
    """
    Generates and saves a 2D scatter plot of the paper clusters.

    Args:
        reduced_embeddings (np.ndarray): 2D coordinates of the papers.
        labels (np.ndarray): Cluster label for each paper.
        n_clusters (int): The number of clusters used.
        cluster_names (list[str]): The generated meaningful names for the clusters.
    """
    print("Generating cluster plot...")
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap='viridis',
        s=60,
        alpha=0.8
    )

    plt.title(f'UMAP Projection of Favorite Papers ({n_clusters} Clusters)', fontsize=20)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if n_clusters > 1:
        legend_elements = scatter.legend_elements()
        plt.legend(legend_elements[0], cluster_names, loc='best', title="Clusters", fontsize=20, title_fontsize=22)

    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
    
    plot_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'cluster_visualization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Cluster plot saved to: {plot_path}")
    return plot_path

def generate_word_cloud(papers):
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

    plt.figure(figsize=(16, 10))
    plt.title('Favorite Papers Word Cloud', fontsize=20)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
        
    wordcloud_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'word_cloud.png')
    plt.savefig(wordcloud_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Word cloud saved to: {wordcloud_path}")
    return wordcloud_path

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
        labels = kmeans.fit_predict(embeddings)
    else:
        labels = np.zeros(len(papers), dtype=int)

    # Generate meaningful names for clusters
    cluster_names = get_cluster_names(papers, labels, n_clusters) if n_clusters > 1 else ["All Papers"]

    plot_path = None
    reducer = None
    reduced_embeddings = None
    n_neighbors = min(15, len(papers) - 1)
    if n_neighbors >= 2:
        print("\n--- Reducing dimensionality with UMAP... ---")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        plot_path = plot_clusters(reduced_embeddings, labels, n_clusters, cluster_names)
    else:
        print("\n--- Not enough papers to perform dimensionality reduction. Skipping cluster plot. ---")

    print("\n--- Generating Word Cloud... ---")
    wordcloud_path = generate_word_cloud(papers)

    return plot_path, wordcloud_path, reducer, reduced_embeddings, labels

def generate_daily_plot(rec_embeddings):
    """
    Generates a daily cluster plot that overlays new recommendations on the base
    cluster map of favorite papers.
    """
    print("\n--- Generating Daily Cluster Plot with Recommendations ---")
    if not rec_embeddings:
        print("Warning: No recommendation embeddings provided. Skipping daily map generation.")
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
        base_papers = cached_data.get('papers')
        
        if reducer is None or base_embeddings_2d is None or base_labels is None or base_papers is None:
            print("Cache is missing necessary UMAP data. Run 'arxiv_engine.py update' to generate it.")
            return None
    except Exception as e:
        print(f"Error loading UMAP data from cache: {e}")
        return None

    # Project new recommendations into the existing 2D space
    print("Projecting new recommendations into the UMAP space...")
    rec_embeddings_2d = reducer.transform(np.array(rec_embeddings))

    # Get cluster names for the legend
    n_clusters = len(np.unique(base_labels))
    cluster_names = get_cluster_names(base_papers, base_labels, n_clusters) if n_clusters > 1 else ["All Papers"]

    # --- Plotting ---
    plt.figure(figsize=(16, 10))
    
    # Plot base favorite papers
    base_scatter = plt.scatter(
        base_embeddings_2d[:, 0], base_embeddings_2d[:, 1], c=base_labels,
        cmap='viridis', s=60, alpha=0.5
    )

    # Plot new recommendations
    rec_scatter = plt.scatter(
        rec_embeddings_2d[:, 0], rec_embeddings_2d[:, 1], c='red', marker='*',
        s=300, edgecolor='black', label='Today\'s Recommendations', zorder=10
    )

    # Annotate new recommendations with their rank
    for i, pos in enumerate(rec_embeddings_2d):
        plt.text(pos[0] + 0.05, pos[1] + 0.05, str(i + 1), color='black', fontsize=12, weight='bold', zorder=11)

    plt.title('Daily Recommendations on Favorite Papers Map', fontsize=20)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- Create a single, combined legend ---
    all_handles = []
    all_labels = []

    # Add handles and labels from the base favorite clusters
    if n_clusters > 1:
        base_handles, _ = base_scatter.legend_elements()
        all_handles.extend(base_handles)
        all_labels.extend(cluster_names)
    
    # Add handle and label for the recommendations
    all_handles.append(rec_scatter)
    all_labels.append('Today\'s Recommendations')

    # Create the combined legend
    plt.legend(all_handles, all_labels, loc='best', title="Legend", fontsize=20, title_fontsize=22)

    # Save plot
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
    
    plot_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'daily_cluster_map.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Daily cluster plot saved to: {plot_path}")
    return plot_path

def main():
    """Main function to run the analysis pipeline as a standalone script."""
    # Load data directly from cache
    papers, embeddings = load_and_process_papers()
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