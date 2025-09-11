# arxiv_daily

## Project Overview

`arxiv_daily` is a personalized, automated tool that acts as your personal research assistant. It fetches the latest papers from arXiv, scores them based on your Zotero library, and generates a beautiful, daily HTML digest. This digest includes not only paper recommendations but also a visual analysis of your research interests, including topic clusters and a word cloud.

---

## Features

- **Zotero API Integration**: Learns your research interests directly and automatically from your Zotero library.
- **Semantic Recommendations**: Uses a state-of-the-art language model (`allenai-specter`) to score new papers based on semantic similarity to your existing library.
- **Automated TLDRs**: Generates a concise, one-sentence summary for each recommended paper using the `sumy` library.
- **Robust Paper Fetching**: Uses the official `arxiv` API to reliably fetch the latest papers, with error handling for API quirks.
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

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/chuam/arxiv_daily.git
    cd arxiv_daily
    ```

2.  **Install Dependencies** (Python 3.10+ is recommended)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Zotero API Access**
    -   Create a Zotero API key: `https://www.zotero.org/settings/keys/new`
    -   Find your User ID on the same page.
    -   Create and edit `config.ini`:
    ```bash
    cp config.ini.template config.ini
    # Now edit config.ini with your credentials
    ```
    *(Optional: You can also specify a Zotero collection ID in `config.ini` to focus the analysis on a specific subset of your library.)*

### 2. Usage

#### Step 1: Build Your Interest Model
The first time you run the tool, or whenever you want to update your interest profile based on your latest Zotero library, run:
```bash
python arxiv_engine.py update
```
This command connects to Zotero, builds your interest models, and automatically runs the analysis to generate plots of your interests. It will intelligently skip the process if no changes are detected in your Zotero library.

#### Step 2: Generate Daily Recommendations
This is the command you will run daily:
```bash
python arxiv_rank.py
```
It fetches the latest arXiv papers published on the previous day, scores them against your profile, and generates a new HTML page.

#### (Optional) Run Standalone Analysis
To re-run only the analysis part (e.g., to try a different number of clusters) without fetching from Zotero, use:
```bash
python analyze_favorites.py [number_of_clusters]
```

---

## Automation & Online Browsing

### GitHub Actions Automation

This project includes a GitHub Actions workflow that runs daily, commits all changes, and deploys HTML pages to the `gh-pages` branch.
To enable Zotero integration in GitHub Actions, you must add the following secrets to your repository's settings (`Settings > Secrets and variables > Actions`):
-   `ZOTERO_USER_ID`: Your Zotero user ID.
-   `ZOTERO_API_KEY`: Your Zotero API key.

The workflow uses `actions/cache` to cache the generated `arxiv_cache.pickle` and `nltk_data/` files between runs, speeding up the process when your Zotero library or dependencies haven't changed. These caches are not committed to the repository.

### GitHub Pages

Enable GitHub Pages in your repository settings, using the `gh-pages` branch as the source.  
You can then browse daily recommendations at:
`https://<your-username>.github.io/arxiv_daily/`

---

## Project Structure

- `arxiv_engine.py`: Core engine for fetching, scoring, and recommending papers. Also handles updating the interest model.
- `arxiv_rank.py`: Main script to generate the daily HTML recommendation page.
- `analyze_favorites.py`: A tool to analyze and visualize your favorite papers. Can be run standalone for advanced analysis.
- `config.py`: Central configuration file for shared constants and settings.
- `config.ini.template`: Template for Zotero API configuration.
- `requirements.txt`: Dependency list.
- `arxiv_history/`: Archive of historical HTML pages.
- `analysis_results/`: Directory for storing generated analysis plots (word cloud, cluster map).
- `.github/workflows/arxiv_daily.yml`: Automation workflow configuration.

---

## FAQ

- **How do I customize my interests?**  
  Simply add relevant papers to your Zotero library. The system will automatically detect changes on the next `update` run. To focus on a specific area, you can provide a Zotero collection ID in `config.ini`.

- **How do I change the categories fetched?**  
  Edit the `ARXIV_CATEGORIES` list in `config.py`.

---

## License & Acknowledgements

This project is based on the original work by Lijing Shao and is licensed under the GPL.
