"""
A dedicated client for interacting with the Zotero API.

This module encapsulates all the logic for fetching and processing data
from a Zotero library, including handling configuration and caching.
"""
import os
import configparser
from pyzotero import zotero
from config import CONFIG_FILE

class ZoteroClient:
    """A client to handle Zotero API interactions."""

    def __init__(self):
        """Initializes the ZoteroClient."""
        self.config = self._load_config()
        self.client = self._get_zotero_client()
        self.favorite_papers = None  # Cache for favorite papers

    def _load_config(self):
        """Loads configuration from config.ini."""
        config = configparser.ConfigParser()
        if not os.path.exists(CONFIG_FILE):
            print("Warning: config.ini not found. Zotero integration will be disabled.")
            return None
        config.read(CONFIG_FILE)
        return config

    def _get_zotero_client(self):
        """Initializes and returns a Zotero client if config is valid."""
        if not self.config or 'zotero' not in self.config:
            return None
        
        try:
            user_id = self.config.get('zotero', 'user_id')
            api_key = self.config.get('zotero', 'api_key')
            if not user_id or not api_key or 'YOUR' in user_id or 'YOUR' in api_key:
                raise ValueError("Zotero user_id or api_key is not set in config.ini")
            return zotero.Zotero(user_id, 'user', api_key)
        except (configparser.NoOptionError, configparser.NoSectionError, ValueError) as e:
            print(f"Warning: Zotero config is incomplete or invalid in config.ini: {e}")
            return None

    def get_last_modified_version(self):
        """
        Fetches the last modified version number of the Zotero library.
        Returns -1 if the client is not configured or an error occurs.
        """
        if not self.client:
            return -1
        try:
            return self.client.last_modified_version()
        except Exception as e:
            print(f"Warning: Could not check Zotero library version: {e}. Proceeding with full update.")
            return -1

    def get_favorite_papers(self):
        """Fetches and caches favorite papers from the Zotero API."""
        if self.favorite_papers is not None:
            return self.favorite_papers

        if not self.client:
            print("Zotero client not configured. Cannot fetch favorite papers.")
            self.favorite_papers = []
            return self.favorite_papers

        collection_id = self.config.get('zotero', 'collection_id', fallback=None)
        
        print(f"Fetching items from Zotero library...")
        try:
            if collection_id:
                print(f"Using collection ID: {collection_id}")
                items = self.client.collection_items(collection_id)
            else:
                print("Fetching all items from the library.")
                items = self.client.everything(self.client.items())
        except Exception as e:
            print(f"Error fetching from Zotero API: {e}")
            self.favorite_papers = []
            return self.favorite_papers
        
        print(f"Found {len(items)} items in Zotero.")

        papers = []
        for item in items:
            data = item.get('data', {})
            if data.get('itemType') not in ['journalArticle', 'conferencePaper', 'preprint', 'report', 'thesis', 'bookSection', 'book']:
                continue

            title = data.get('title', '')
            abstract = data.get('abstractNote', '')
            authors_list = [f"{c.get('firstName', '')} {c.get('lastName', '')}".strip() for c in data.get('creators', []) if c.get('creatorType') == 'author']
            
            date_added = data.get('dateAdded') # Get the dateAdded field
            
            if title and authors_list:
                papers.append({
                    'key': item['key'], 
                    'version': data['version'], 
                    'title': title, 
                    'abstract': abstract, 
                    'authors': authors_list,
                    'dateAdded': date_added # Add it to the paper dict
                })
        
        self.favorite_papers = papers
        return self.favorite_papers