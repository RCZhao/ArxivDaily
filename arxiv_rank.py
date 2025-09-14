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
from arxiv_paper import ArxivPaper
from analysis import generate_daily_plot, generate_score_distribution_plot
from config import BASE, MAX_PAPERS_TO_SHOW, MIN_SCORE_THRESHOLD, AUTHOR_COLLAPSE_THRESHOLD, ANALYSIS_OUTPUT_DIR

def generate_page(recommended_papers, out_file_name, page_title_date, cluster_plot_path=None, word_cloud_path=None, score_dist_plot_path=None):
    """Generates and writes the final HTML page using a Jinja2 template.

    Args:
        recommended_papers (list[ArxivPaper]): A list of recommended paper objects.
        out_file_name (str): The path to the output HTML file.
        page_title_date (str): The date string for the page title.
        cluster_plot_path (str, optional): Path to the cluster visualization image.
        word_cloud_path (str, optional): Path to the word cloud image.
        score_dist_plot_path (str, optional): Path to the score distribution plot.
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
        'page_title': f'arXiv Daily: {page_title_date}',
        'current_date': page_title_date,
        'current_year': time.strftime("%Y"),
        'cluster_plot_path': relative_cluster_path,
        'word_cloud_path': relative_wordcloud_path,
        'score_dist_plot_path': relative_score_dist_plot_path,
        'recommended_papers': recommended_papers,
        'author_collapse_threshold': AUTHOR_COLLAPSE_THRESHOLD,
    }

    # --- Render and save the page ---
    with open(out_file_name, "w", encoding='utf-8') as f:
        html_content = template.render(context)
        f.write(html_content)

    print(f'''\nIn total {len(recommended_papers)} articles for your preview; please enjoy and have
          a good day!\n''')
    print(f"HTML page saved to: {out_file_name}")


def run_daily_rank(algorithm):
    """Runs the standard daily recommendation generation."""
    print(f"Using '{algorithm}' scoring algorithm.")
    
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
    if recommendations:
        try:
            # Extract embeddings and labels for the plot
            rec_embeddings = [paper.embedding for paper in recommendations]
            rec_labels = [paper.cluster_id for paper in recommendations]
            daily_plot_path = generate_daily_plot(rec_embeddings, rec_labels)
        except Exception as e:
            print(f"Warning: Failed to generate daily cluster plot. Error: {e}")
    
    # Use the new daily plot if available, otherwise fall back to the static one
    final_cluster_plot = daily_plot_path if daily_plot_path else cluster_plot
    if not daily_plot_path and recommendations:
        print("\nWarning: Could not generate daily recommendation map. Falling back to static cluster map.")
        print("         This is likely because the cache is missing UMAP data.")
        print("         Please run 'python arxiv_engine.py update' to generate it.")
    
    # Generate the final HTML page including analysis results
    out_file_name = os.path.join(BASE, time.strftime("%Y%m%d") + ".html")
    page_date = time.strftime("%Y-%m-%d")
    generate_page(
        recommendations,
        out_file_name=out_file_name,
        page_title_date=page_date,
        cluster_plot_path=final_cluster_plot,
        word_cloud_path=word_cloud,
        score_dist_plot_path=score_dist_plot
    )

def run_backfill(start_date, end_date, algorithm):
    """Runs a historical backfill for a given date range."""
    page_title_date_str = f"Backfill ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
    print(f"--- Starting Historical Backfill ---")
    print(f"Period: {page_title_date_str}")
    print(f"Using '{algorithm}' scoring algorithm.")

    recommender = ArxivEngine(mode='feed')

    # Get historical recommendations
    recommendations, all_scored_papers = recommender.get_historical_recommendations(
        start_date=start_date,
        end_date=end_date,
        max_papers=MAX_PAPERS_TO_SHOW,
        min_score=MIN_SCORE_THRESHOLD,
        algorithm=algorithm
    )

    # Generate score distribution plot
    score_dist_plot = generate_score_distribution_plot(all_scored_papers, MIN_SCORE_THRESHOLD)

    # Generate the cluster plot with backfill recommendations
    backfill_plot_path = None
    if recommendations:
        try:
            rec_embeddings = [paper.embedding for paper in recommendations]
            rec_labels = [paper.cluster_id for paper in recommendations]
            backfill_plot_path = generate_daily_plot(
                rec_embeddings, 
                rec_labels,
                output_filename='backfill_cluster_map.png',
                title='Backfill Recommendations on Favorite Papers Map'
            )
        except Exception as e:
            print(f"Warning: Failed to generate backfill cluster plot. Error: {e}")

    # For backfill, we use the newly generated plot, or fall back to static plots.
    final_cluster_plot = backfill_plot_path
    if not backfill_plot_path:
        print("\nWarning: Could not generate backfill recommendation map. Using static plots if available.")
        final_cluster_plot = os.path.join(ANALYSIS_OUTPUT_DIR, 'cluster_visualization.png')
        if not os.path.exists(final_cluster_plot):
            final_cluster_plot = None

    word_cloud = os.path.join(ANALYSIS_OUTPUT_DIR, 'word_cloud.png')
    if not os.path.exists(word_cloud):
        word_cloud = None

    # Generate the final HTML page
    out_file_name = os.path.join(BASE, f"backfill_{time.strftime('%Y%m%d')}.html")
    generate_page(
        recommendations,
        out_file_name=out_file_name,
        page_title_date=page_title_date_str,
        cluster_plot_path=final_cluster_plot,
        word_cloud_path=word_cloud,
        score_dist_plot_path=score_dist_plot
    )

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

    # Create a mutually exclusive group for date selection to avoid conflicts.
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        '--start-date',
        type=str,
        help="Start date for historical backfill in YYYY-MM-DD format."
    )
    date_group.add_argument(
        '--days',
        type=int,
        help="Run a historical backfill for the last N days."
    )
    date_group.add_argument(
        '--weeks',
        type=int,
        help="Run a historical backfill for the last N weeks."
    )
    date_group.add_argument(
        '--months',
        type=int,
        help="Run a historical backfill for the last N months."
    )
    date_group.add_argument(
        '--years',
        type=int,
        help="Run a historical backfill for the last N years."
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help="End date for historical backfill in YYYY-MM-DD format. Defaults to today. Can be used with --start-date."
    )

    args = parser.parse_args()
    
    # Determine if we are in backfill mode
    is_backfill = any([args.start_date, args.days, args.weeks, args.months, args.years])

    if is_backfill:
        from datetime import datetime, timedelta, timezone
        end_dt = datetime.now(timezone.utc)
        if args.end_date:
            try:
                end_dt = datetime.strptime(args.end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            except ValueError:
                parser.error("Invalid end-date format. Please use YYYY-MM-DD.")

        if args.start_date:
            try:
                start_dt = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            except ValueError:
                parser.error("Invalid start-date format. Please use YYYY-MM-DD.")
        elif args.days:
            start_dt = end_dt - timedelta(days=args.days)
        elif args.weeks:
            start_dt = end_dt - timedelta(weeks=args.weeks)
        elif args.months:
            start_dt = end_dt - timedelta(days=int(args.months * 30.44))
        elif args.years:
            start_dt = end_dt - timedelta(days=int(args.years * 365.25))
        
        if start_dt >= end_dt:
            parser.error("Start date must be before end date.")

        run_backfill(start_date=start_dt, end_date=end_dt, algorithm=args.algorithm)
    else:
        if args.end_date:
            parser.error("--end-date can only be used with --start-date or a relative period like --days.")
        run_daily_rank(algorithm=args.algorithm)
