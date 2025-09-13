"""
This module defines the ArxivPaper class, which represents a single arXiv paper
and its associated metadata and scores.
"""
import re
import arxiv

class ArxivPaper:
    """Represents a single arXiv paper and its associated metadata and scores."""

    # Create a translation table for cleaning text for embedding.
    _PUNCTUATION_TO_REPLACE = '''$,'.()[]{}?!&*/\`~<>^%-+|"'''
    _TRANSLATION_TABLE = str.maketrans(_PUNCTUATION_TO_REPLACE, ' ' * len(_PUNCTUATION_TO_REPLACE))

    def __init__(self, arxiv_result):
        """
        Initializes an ArxivPaper object from an arxiv.Result object or a feedparser entry.

        Args:
            arxiv_result (arxiv.Result or dict): The paper data.
        """
        if isinstance(arxiv_result, arxiv.Result):
            self.entry_id = arxiv_result.entry_id
            self.raw_title = f"{arxiv_result.title} (arXiv:{arxiv_result.get_short_id()})"
            self.raw_summary = arxiv_result.summary
            self.authors = [author.name for author in arxiv_result.authors]
            self.primary_category = arxiv_result.primary_category
        else: # Handle feedparser-like dict for backward compatibility if needed
            self.entry_id = arxiv_result.get('link', '')
            self.raw_title = arxiv_result.get('title', '')
            self.raw_summary = arxiv_result.get('summary', '')
            self.authors = [a.strip() for a in arxiv_result.get('author', '').split(',')]
            self.primary_category = arxiv_result.get('category', '')

        # Processed data for display
        self.arxiv_id = self._extract_arxiv_id()
        self.title_text = self._extract_title_text()
        self.status = self._determine_status()
        self.cleaned_summary = ArxivPaper._clean_text(self.raw_summary, for_embedding=False)

        # Processed data for embedding, stored on the object
        self.title_for_embedding = ArxivPaper._clean_text(self.title_text, for_embedding=True)
        self.summary_for_embedding = ArxivPaper._clean_text(self.raw_summary, for_embedding=True)

        # Data to be filled in by the engine
        self.tldr = ""
        self.score = 0.0
        self.score_list = [0.0, 0.0]
        self.embedding = None
        self.cluster_id = None
        self.cluster_name = ""

    def _extract_arxiv_id(self):
        match = re.search(r'(\d+\.\d+)', self.entry_id)
        return match.group(1) if match else ""

    def _extract_title_text(self):
        match = re.match(r"^(.*)\s+\(arXiv:(\d+\.\d+)(?:v\d+)?\)", self.raw_title)
        return match.group(1).strip() if match else self.raw_title.strip()

    def _determine_status(self):
        summary_lower = self.raw_summary.lower()
        if 'announce type: replace' in summary_lower:
            return 'replace'
        elif 'announce type: cross' in summary_lower:
            return 'cross-list'
        return ''

    @staticmethod
    def _clean_text(text, for_embedding=False):
        """
        Cleans text for display or for embedding.

        Args:
            text (str): The input string.
            for_embedding (bool): If True, applies cleaning specific for embedding
                                  (lowercasing, punctuation removal).

        Returns:
            str: The cleaned text.
        """
        # Common cleaning: remove HTML tags and "Abstract:" prefix
        cleaned_text = re.sub(r'<[^>]*>', '', text)
        cleaned_text = re.sub(r'.*?Abstract:\s*', '', cleaned_text, count=1, flags=re.IGNORECASE | re.DOTALL)

        if for_embedding:
            cleaned_text = cleaned_text.lower()
            cleaned_text = cleaned_text.translate(ArxivPaper._TRANSLATION_TABLE)

        # Final whitespace normalization
        return ' '.join(cleaned_text.split())