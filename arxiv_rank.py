"""Main script for generating the daily arXiv recommendation HTML page.

This script acts as the presentation layer. It uses the ArxivEngine to get a
list of recommended papers and then formats them into a user-friendly HTML
page that is automatically opened in a browser.
"""
import sys
import time
import re
import os
import numpy as np

from arxiv_engine import ArxivEngine
from analyze_favorites import generate_daily_plot
from config import BASE, MAX_PAPERS_TO_SHOW, MIN_SCORE_THRESHOLD, AUTHOR_COLLAPSE_THRESHOLD, ANALYSIS_OUTPUT_DIR


#---------------------------------------------------------------------
MATHJAX = '''
<style>
/* Improved abstract readability */
.arxiv-abstract-text {
    font-size: 1.1em;
    line-height: 1.7;
}
p.big {
    line-height: 1.6;
    font-size: large;
}
</style>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'] ],
      processEscapes: true
    }
  });
</script>
<script>
function toggleAuthors(element) {
    const fullList = element.querySelector('.author-full');
    const shortList = element.querySelector('.author-short');
    const toggleText = element.querySelector('.author-toggle');
    
    fullList.style.display = fullList.style.display === 'none' ? 'inline' : 'none';
    shortList.style.display = shortList.style.display === 'none' ? 'inline' : 'none';
    toggleText.textContent = fullList.style.display === 'none' ? ' (show all)' : ' (show less)';
}
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>'''
#---------------------------------------------------------------------

def format_entry_to_html(index, item, cluster_names=None):
    """Formats a single scored paper item into an HTML card.

    Args:
        item (dict): A dictionary containing the paper's score, score components,
                     and the original feedparser entry.

    Returns:
        str: An HTML string representing a Bootstrap card for the paper.
    """
    score = item['score']
    score_list = item['score_list']
    tldr = item.get('tldr', '') # Safely get TLDR
    entry = item['entry']
    cluster_id = item.get('cluster_id')
    category = entry.get('category')

    # Use regex to extract title and arxiv id
    match = re.match(r"^(.*)\s+\(arXiv:(\d+\.\d+)(?:v\d+)?\)", entry['title'])
    if match:
        title_text = match.group(1)
        arxiv_id = match.group(2)
    else:
        title_text = entry['title']
        arxiv_id = ""

    # Check for update/cross-list status from summary
    status = ""
    summary_lower = entry['summary'].lower()
    if 'announce type: replace' in summary_lower:
        status = 'replace'
    elif 'announce type: cross' in summary_lower:
        status = 'cross-list'

    if status:
        title_text += f' <span class="badge bg-warning text-dark">{status}</span>'

    # Handle long author lists
    authors_html = f"<strong>{entry['author']}</strong>"
    author_list = entry['author'].split(', ')
    if len(author_list) > AUTHOR_COLLAPSE_THRESHOLD:
        short_author_list = ', '.join(author_list[:AUTHOR_COLLAPSE_THRESHOLD]) + ', et al.'
        authors_html = f'''
            <div onclick="toggleAuthors(this)" style="cursor: pointer;">
                <strong>
                    <span class="author-short">{short_author_list}</span>
                    <span class="author-full" style="display: none;">{entry['author']}</span>
                </strong>
                <a class="author-toggle text-primary" style="text-decoration: none;"> (show all)</a>
            </div>
        '''


    links = f'''
        <div class="arxiv-links mt-2">
            <a class="btn btn-sm btn-outline-primary" href="https://arxiv.org/abs/{arxiv_id}" target="_blank">arXiv</a>
            <a class="btn btn-sm btn-outline-success" href="https://arxiv.org/pdf/{arxiv_id}" target="_blank">PDF</a>
            <a class="btn btn-sm btn-outline-secondary" href="https://ui.adsabs.harvard.edu/#abs/arXiv:{arxiv_id}" target="_blank">ADS</a>
        </div>
    '''
    scores = f'''
        <div class="arxiv-scores">
            <span class="badge bg-success">Total Score: {score:.1f}</span>
            <span class="badge bg-info text-dark">Author Score: {score_list[0]:.2f}</span>
            <span class="badge bg-primary">Semantic Score: {score_list[1]:.2f}</span>
    '''
    if category:
        scores += f' <span class="badge bg-secondary">{category}</span>'
    
    if cluster_id is not None and cluster_names:
        cluster_name = cluster_names[cluster_id] if cluster_id < len(cluster_names) else f"Cluster {cluster_id}"
        # A purple-ish color for the interest cluster badge
        scores += f' <span class="badge" style="background-color: #6f42c1; color: white;">Interest: {cluster_name}</span>'

    scores += '</div>'
    # Clean abstract for display: remove HTML tags and normalize whitespace
    abstract = re.sub(r'<[^>]*>', '', entry['summary'])
    abstract = re.sub(r'\s+', ' ', abstract).strip()
    # Defensively remove any "Abstract: " or similar prefixes that might have been added
    # This handles cases where the summary might be polluted with extra metadata.
    # This finds the first instance of "Abstract: " and takes everything after it.
    abstract = re.sub(r'.*?Abstract:\s*', '', abstract, count=1, flags=re.IGNORECASE | re.DOTALL).strip()
    html = f'''
    <div class="arxiv-card card mb-4">
        <div class="arxiv-title card-header">{index}. {title_text}</div>
        <div class="arxiv-abstract card-body">
            <div class="arxiv-meta mb-3">{authors_html}</div>
            <div class="tldr-section mb-3">
                <p class="fst-italic text-muted"><strong>TLDR:</strong> {tldr}</p>
            </div>
            <div class="arxiv-abstract-text">{abstract}</div>
            <div class="mt-3">{links}</div>
            <div class="mt-2">{scores}
        </div>
    </div>
    '''
    return html


def generate_page(recommended_papers, cluster_plot_path=None, word_cloud_path=None, cluster_names=None):
    """Generates and writes the final HTML page.

    This function creates the daily recommendation page, including an optional
    section for the analysis of favorite papers if the plot paths are provided.

    The page is saved with the current date (e.g., '20231027.html') and then
    opened in the default system web browser.

    Args:
        recommended_papers (list[dict]): A list of recommended paper items,
            as returned by ArxivEngine.get_recommendations().
        cluster_plot_path (str, optional): Path to the cluster visualization image.
        word_cloud_path (str, optional): Path to the word cloud image.
    """
    browser_cmd = "open"
    out_file_name = os.path.join(BASE, time.strftime("%Y%m%d") + ".html")

    with open(out_file_name, "w") as f:
        f.write('<html lang="en">\n')
        f.write('''<head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>arXiv daily</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
            <style>
                body { font-family: 'Roboto', Arial, sans-serif; background: #f8f9fa; transition: background 0.3s; }
                .arxiv-card { margin: 2em auto; max-width: 900px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-radius: 16px; transition: box-shadow 0.3s, transform 0.3s; }
                .arxiv-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); transform: translateY(-1px); }
                .arxiv-title { background: linear-gradient(90deg, #0d6efd 60%, #6610f2 100%); color: #fff; border-radius: 16px 16px 0 0; padding: 1.5em; font-size: 1.5em; font-weight: 700; letter-spacing: 0.5px;}
                .arxiv-meta { font-size: 1em; color: #555; margin-bottom: 1em; }
                .arxiv-abstract { background: #fff; padding: 1.5em; border-radius: 0 0 16px 16px; font-size: 1.1em; line-height: 1.7;}
                .arxiv-links a { margin-right: 1em; }
                .arxiv-scores { margin-top: 1em; font-size: 1em; }
                .navbar { background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
                .footer { text-align: center; color: #888; padding: 2em 0 1em 0; font-size: 0.95em;}
                @media (prefers-color-scheme: dark) {
                    body { background: #181a1b; color: #e4e6eb; }
                    .arxiv-card { background: #23272b; box-shadow: 0 2px 8px rgba(0,0,0,0.28);}
                    .arxiv-card:hover { box-shadow: 0 5px 15px rgba(0,0,0,0.25); transform: translateY(-1px); }
                    .arxiv-title { background: linear-gradient(90deg, #375a7f 60%, #6f42c1 100%);}
                    .arxiv-abstract { background: #23272b; color: #e4e6eb;}
                    .navbar { background: #23272b; color: #e4e6eb;}
                    .text-muted { color: #adb5bd !important; }
                }
            </style>
            ''' + MATHJAX + '</head>\n')
        f.write('<body>\n')
        f.write('''
        <nav class="navbar navbar-expand-lg mb-4">
            <div class="container">
                <a class="navbar-brand fw-bold text-primary" href="#">arXiv Daily</a>
                <span class="navbar-text">Modern arXiv Reader</span>
            </div>
        </nav>
        ''')

        # Add analysis section if plots are available
        if cluster_plot_path and word_cloud_path:
            relative_cluster_path = os.path.relpath(cluster_plot_path, BASE)
            relative_wordcloud_path = os.path.relpath(word_cloud_path, BASE)
            f.write(f'''
            <div class="container py-4">
                <header class="mb-4"><h2 class="display-6 text-center">Analysis of Your Favorites</h2></header>
                <div class="row align-items-center">
                    <div class="col-lg-6 mb-4">
                        <div class="card h-100">
                            <div class="card-header fw-bold">Interest Word Cloud</div>
                            <div class="card-body text-center p-2">
                                <img src="{relative_wordcloud_path}" class="img-fluid rounded" alt="Word Cloud of favorite paper topics">
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-6 mb-4">
                        <div class="card h-100">
                            <div class="card-header fw-bold">Interest Clusters</div>
                            <div class="card-body text-center p-2">
                                <img src="{relative_cluster_path}" class="img-fluid rounded" alt="2D Cluster plot of favorite papers">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            ''')

        f.write('<div class="container py-4">\n')
        f.write('<header class="mb-4"><h1 class="display-5 text-center text-primary">arXiv:'
                + time.strftime("%Y-%m-%d") + '</h1></header>\n')

        for i, item in enumerate(recommended_papers, 1):
            f.write(format_entry_to_html(i, item, cluster_names=cluster_names))

        f.write('</div>\n')
        f.write('''
        <footer class="footer">
            &copy; {year} arXiv Daily · Powered by <a href="https://arxiv.org/" target="_blank">arXiv</a>
        </footer>
        '''.format(year=time.strftime("%Y")))
        f.write('</body></html>\n')

    # os.system(browser_cmd + " " + out_file_name)
    print(f'''\nIn total {len(recommended_papers)} articles for your preview; please enjoy and have
          a good day!\n''')


if __name__ == '__main__':
    # This block handles the main execution logic.
    # It can be run with 'update' to show a help message, or with no arguments
    # to generate the daily feed.
    if len(sys.argv) > 1:
        if sys.argv[1] == 'update':
            print("To update the model, please run: python arxiv_engine.py update")
            sys.exit()
    else:
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
        recommendations = recommender.get_recommendations(MAX_PAPERS_TO_SHOW, MIN_SCORE_THRESHOLD)
        
        # Generate the daily plot with recommendations
        daily_plot_path = None
        cluster_names_for_page = None
        if recommendations:
            try:
                # Extract embeddings and labels for the plot
                rec_embeddings = [item['embedding'] for item in recommendations]
                rec_labels = [item['cluster_id'] for item in recommendations]
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
        generate_page(recommendations, cluster_plot_path=final_cluster_plot, word_cloud_path=word_cloud, cluster_names=cluster_names_for_page)
