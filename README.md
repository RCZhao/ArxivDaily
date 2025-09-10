# arxiv_daily

## Project Overview

`arxiv_daily` is a personalized, automated tool that acts as your personal research assistant. It fetches the latest papers from arXiv, scores them based on your Zotero library, and generates a beautiful, daily HTML digest. This digest includes not only paper recommendations but also a visual analysis of your research interests, including topic clusters and a word cloud.

<!-- You can replace this with a real screenshot of your generated HTML page -->
<!--  -->

---

## Features

- **Zotero API Integration**: Learns your research interests directly and automatically from your Zotero library.
- **Semantic Recommendations**: Uses a state-of-the-art language model (`allenai-specter`) to score new papers based on semantic similarity to your existing library.
- **Automated TLDRs**: Generates a concise, one-sentence summary for each recommended paper.
- **Intelligent Interest Analysis**:
   - Automatically determines the optimal number of research clusters in your library using the Silhouette Score.
   - Generates meaningful keyword labels for each topic cluster using TF-IDF.
   - Creates a 2D visualization of your interest clusters and a word cloud of key terms.
- **Daily HTML Digest**: Produces a clean, modern, and interactive HTML page with your daily recommendations and interest analysis.
- **Fully Automated**: Includes a GitHub Actions workflow that runs the entire pipeline daily, commits the results, and deploys the page to GitHub Pages.
- **Efficient Caching**: Intelligently checks for changes in your Zotero library to avoid unnecessary re-computation, making updates fast and efficient.

---

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/chuam/arxiv_daily.git
cd arxiv_daily

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Dependencies

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

### 3. Run Locally

1.  **Set up Zotero API Access**
    -   Create a Zotero API key from your Zotero account settings: `https://www.zotero.org/settings/keys/new`
    -   Find your User ID in the same settings page (`https://www.zotero.org/settings/keys`).
    -   Create a `config.ini` file by copying `config.ini.template` and fill in your `user_id` and `api_key`.
    ```bash
    cp config.ini.template config.ini
    # Now edit config.ini with your credentials
    ```

2.  **Generate Daily Recommendations**
    ```bash
    python arxiv_rank.py
    ```
    This generates today's HTML recommendation page.

3.  **(Optional) Update Interest Model**  
    The interest model is built from your Zotero library. Run this command to build or update it:
    ```bash
    python arxiv_engine.py update
    ```

4.  **(Optional) Analyze Your Favorites**  
    To get a deeper insight into your collected papers, run the analysis script:
    ```bash
    python analyze_favorites.py
    ```
    This will create a `cluster_visualization.png` and `word_cloud.png` in the `analysis_results/` directory.

---

## Analysis of Your Interests

The `analyze_favorites.py` script helps you understand your research interests based on the papers in your Zotero library.

### Features

-   **Cluster Analysis**: Groups your favorite papers into distinct topics.
-   **Visualization**: Creates a 2D plot showing how different papers and clusters relate to each other.
-   **Word Cloud**: Generates a word cloud of the most frequent terms in your favorite papers' titles and abstracts.


## Automation & Online Browsing

### GitHub Actions Automation

This project includes a GitHub Actions workflow that runs daily, commits all changes, and deploys HTML pages to the `gh-pages` branch.
To enable Zotero integration in GitHub Actions, you must add the following secrets to your repository's settings (`Settings > Secrets and variables > Actions`):
-   `ZOTERO_USER_ID`: Your Zotero user ID.
-   `ZOTERO_API_KEY`: Your Zotero API key.


The workflow uses `actions/cache` to cache the generated `arxiv_cache.pickle` file between runs, speeding up the process when your Zotero library hasn't changed. This cache is not committed to the repository.

Enable GitHub Pages in your repository settings, using the `gh-pages` branch as the source.  
You can then browse daily recommendations at:

```
https://your-username.github.io/your-repo-name/
```

---

## Main Files

- `arxiv_engine.py`: Core engine for fetching, scoring, and recommending papers.
- `arxiv_rank.py`: Main script to generate the daily HTML recommendation page.
- `analyze_favorites.py`: A tool to analyze and visualize your favorite papers.
- `config.ini.template`: Template for Zotero API configuration.
- `requirements.txt`: Dependency list
- `arxiv_history/`: Archive of historical HTML pages
- `.github/workflows/arxiv_daily.yml`: Automation workflow configuration

---

## FAQ

- **How do I customize my interests?**  
  The system learns from your Zotero library. Add relevant papers to your Zotero account. If you want to use a specific collection, you can specify its ID in `config.ini`. Remember to run `python arxiv_engine.py update` after making significant changes to your library.

- **How do I change the categories fetched?**  
  Edit the `RSS_URLS` list in `arxiv_rank.py`.

- **How do I deploy to my own repository?**  
  Fork this project and set up GitHub Pages as described above.

---

## License

GPL

---

Feel free to open an issue or PR if you have questions