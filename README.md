# arxiv_daily

## Project Overview

`arxiv_daily` is an automated tool that fetches the latest papers from various arXiv categories, scores them based on your interests and history, and generates a daily HTML recommendation page. With GitHub Actions, you can automate daily runs and browse the results online via GitHub Pages.

---

## Features

- Automatically fetches the latest papers from arXiv RSS feeds
- Scores papers based on authors, titles, abstracts, and your favorites
- Learns your interests from your favorite arXiv links
- Generates a daily HTML page with scores and abstracts
- Supports automated deployment to GitHub Pages for online browsing
- Archives all data and historical results

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/arxiv_daily.git
cd arxiv_daily
```

### 2. Install Dependencies

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
```

### 3. Run Locally

```bash
python arxiv_rank.py update   # (Optional) Update interest model
python arxiv_rank.py          # Generate today's HTML recommendation page
```

The generated HTML file will be named by date (e.g., `20250909.html`) and copied as `index.html`. Historical files are stored in the `arxiv_history/` folder.

---

## Automation & Online Browsing

### GitHub Actions Automation

This project includes a GitHub Actions workflow that runs daily, commits all changes, and deploys HTML pages to the `gh-pages` branch.

### GitHub Pages Online Browsing

Enable GitHub Pages in your repository settings, using the `gh-pages` branch as the source.  
You can then browse daily recommendations at:

```
https://your-username.github.io/your-repo-name/
```

---

## Main Files

- `arxiv_rank.py`: Core logic for fetching and scoring papers
- `requirements.txt`: Dependency list
- `favorite_arxiv_links.txt`: Your favorite arXiv links
- `arxiv_history/`: Archive of historical HTML pages
- `.github/workflows/arxiv_daily.yml`: Automation workflow configuration

---

## FAQ

- **How do I customize my interests?**  
  Edit `favorite_arxiv_links.txt` and add your favorite arXiv paper links. The model will update your interests on the next run.

- **How do I change the categories fetched?**  
  Edit the `RSS_URLS` list in `arxiv_rank.py`.

- **How do I deploy to my own repository?**  
  Fork this project and set up GitHub Pages as described above.

---

## License

GPL

---

Feel free to open an issue or PR if you have questions