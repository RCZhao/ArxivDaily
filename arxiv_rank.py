"""Main script for generating the daily arXiv recommendation HTML page.

This script acts as the presentation layer. It uses the ArxivEngine to get a
list of recommended papers and then formats them into a user-friendly HTML
page that is automatically opened in a browser.
"""
import argparse
import time
import os
from jinja2 import Environment, FileSystemLoader

from arxiv_engine import ArxivEngine
from analysis import generate_daily_plot, generate_score_distribution_plot
from config import BASE, MAX_PAPERS_TO_SHOW, MIN_SCORE_THRESHOLD, AUTHOR_COLLAPSE_THRESHOLD, ANALYSIS_OUTPUT_DIR

def generate_page(recommended_papers, cluster_plot_path=None, word_cloud_path=None, score_dist_plot_path=None, cluster_names=None):
    """Generates and writes the final HTML page using a Jinja2 template.

    Args:
        recommended_papers (list[ArxivPaper]): A list of recommended paper objects.
        cluster_plot_path (str, optional): Path to the cluster visualization image.
        word_cloud_path (str, optional): Path to the word cloud image.
        score_dist_plot_path (str, optional): Path to the score distribution plot.
        cluster_names (list[str], optional): Names for the interest clusters.
    """
    # --- Setup Jinja2 environment ---
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    template = env.get_template('daily_page.html')

    # --- Prepare template context ---
    relative_cluster_path = os.path.relpath(cluster_plot_path, BASE) if cluster_plot_path else None
    relative_wordcloud_path = os.path.relpath(word_cloud_path, BASE) if word_cloud_path else None
    relative_score_dist_plot_path = os.path.relpath(score_dist_plot_path, BASE) if score_dist_plot_path else None

    context = {
        'page_title': 'arXiv Daily',
        'current_date': time.strftime("%Y-%m-%d"),
        'current_year': time.strftime("%Y"),
        'cluster_plot_path': relative_cluster_path,
        'word_cloud_path': relative_wordcloud_path,
        'score_dist_plot_path': relative_score_dist_plot_path,
        'recommended_papers': recommended_papers,
        'cluster_names': cluster_names,
        'author_collapse_threshold': AUTHOR_COLLAPSE_THRESHOLD,
    }

    # --- Render and save the page ---
    out_file_name = os.path.join(BASE, time.strftime("%Y%m%d") + ".html")
    with open(out_file_name, "w", encoding='utf-8') as f:
        html_content = template.render(context)
        f.write(html_content)

    print(f'''\nIn total {len(recommended_papers)} articles for your preview; please enjoy and have
          a good day!\n''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a daily digest of recommended arXiv papers. To update the interest model, run 'python arxiv_engine.py update'."
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['max_similarity', 'z_score'],
        default='z_score',
        help="The scoring algorithm to use for semantic similarity. 'z_score' (default) "
             "normalizes similarity by cluster dispersion. 'max_similarity' uses the "
             "raw maximum similarity."
    )
    args = parser.parse_args()
    print(f"Using '{args.algorithm}' scoring algorithm.")

    # Initialize the engine
    recommender = ArxivEngine(mode='feed')

    # Check for existing analysis plots to include in the page
    cluster_plot = os.path.join(ANALYSIS_OUTPUT_DIR, 'cluster_visualization.png')
    word_cloud = os.path.join(ANALYSIS_OUTPUT_DIR, 'word_cloud.png')

    if not os.path.exists(cluster_plot) or not os.path.exists(word_cloud):
        print("\nAnalysis plots not found. Skipping analysis section in HTML.")
        cluster_plot = None
        word_cloud = None
    else:
        print("\nFound analysis plots. They will be included in the HTML page.")

    # Get daily recommendations
    print("\n--- Getting Daily Recommendations ---")
    recommendations, all_scored_papers = recommender.get_recommendations(
        MAX_PAPERS_TO_SHOW, MIN_SCORE_THRESHOLD, algorithm=args.algorithm
    )
    
    # Generate score distribution plot for all papers found
    score_dist_plot = generate_score_distribution_plot(all_scored_papers, MIN_SCORE_THRESHOLD)
    
    # Generate the daily plot with recommendations
    daily_plot_path = None
    cluster_names_for_page = None
    if recommendations:
        try:
            # Extract embeddings and labels for the plot
            rec_embeddings = [paper.embedding for paper in recommendations]
            rec_labels = [paper.cluster_id for paper in recommendations]
            daily_plot_path, cluster_names_for_page = generate_daily_plot(rec_embeddings, rec_labels)
        except Exception as e:
            print(f"Warning: Failed to generate daily cluster plot. Error: {e}")
    
    # Use the new daily plot if available, otherwise fall back to the static one
    final_cluster_plot = daily_plot_path if daily_plot_path else cluster_plot
    if not daily_plot_path and recommendations:
        print("\nWarning: Could not generate daily recommendation map. Falling back to static cluster map.")
        print("         This is likely because the cache is missing UMAP data.")
        print("         Please run 'python arxiv_engine.py update' to generate it.")
    
    # Generate the final HTML page including analysis results
    generate_page(recommendations, cluster_plot_path=final_cluster_plot, word_cloud_path=word_cloud, score_dist_plot_path=score_dist_plot, cluster_names=cluster_names_for_page)
