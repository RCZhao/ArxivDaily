""" An arXiv class that learns from arXiv links and produces scores
for arXiv links"""

import feedparser
import requests
from bs4 import BeautifulSoup
from rake_nltk import Rake
import re
import pickle
import os
import sys
import time

__author__ = 'Lijing Shao'
__email__ = 'Friendshao@gmail.com'
__licence__ = 'GPL'

#---------------------------------------------------------------------
BASE = '/users/rzhao/arxiv/'
EXAMPLE_URL = 'https://arxiv.org/abs/1705.01278'

# Overall score weights
TITLE_SCORE = 60.0
ABSTRACT_SCORE = 25.0
AUTHOR_SCORE = 50.0

RSS_URLS = ["http://export.arxiv.org/rss/astro-ph",
            "http://export.arxiv.org/rss/gr-qc",
            "http://export.arxiv.org/rss/hep-ph",
            "http://export.arxiv.org/rss/physics.comp-ph",
            "http://export.arxiv.org/rss/physics.data-an",
            "http://export.arxiv.org/rss/physics.gen-ph",
            "http://export.arxiv.org/rss/physics.hist-ph",
            "http://export.arxiv.org/rss/physics.soc-ph",
            "http://export.arxiv.org/rss/physics.pop-ph",
            "http://export.arxiv.org/rss/math.HO",
            "http://export.arxiv.org/rss/math.PR",
            "http://export.arxiv.org/rss/math.ST",
            "http://export.arxiv.org/rss/stat",
            "http://export.arxiv.org/rss/CoRR"
           ]

MATHJAX = '''
<style>
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
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>'''
#---------------------------------------------------------------------

class arXiv(object):
    """ arXiv class

    methods:
        __init__(self, mode='feed')
        update_score_files(self)
        unique_favorite_arxiv_links(self, verbose=True)
        load_score_files(self)
        get_content_from_url(self, url=EXAMPLE_URL)
        get_content_from_entry(self, entry=dict())
        score_from_url(self, url=EXAMPLE_URL, load_score=True,
                       score_files=(None, None), verbose=False)
        score_from_entry(self, entry=dict(),
                         load_score=True, score_files=(None, None),
                         verbose=False)
        generate_weight_from_url(self, url=EXAMPLE_URL)
        format_entry_to_html(self, entry)
        clean_text(self, text)
    """

    def __init__(self, mode='feed'):
        """ Two modes {'update', 'feed'}
            - 'update': update score files
            - 'feed': obtain daily feed
        """
        self.mode = mode
        if mode == 'update':
            self.favorite_links = self.update_score_files()
        self.author, self.title_abstract = self.load_score_files()

    def clean_text(self, text):
        """ Given a text, clean it """
        for s in '''$,'.()[]{}?!&*/\`~<>^%-+|"''':
            text = text.replace(s, ' ')
        text = [x.strip() for x in text.split(' ') if len(x.strip()) > 0]
        return ' '.join(text)

    def update_score_files(self):
        """ Take new links, merge them into favorite links, and update the
        score files
        """
        links = self.unique_favorite_arxiv_links()
        new_links = list(set(open(BASE+'update_arxiv_links.txt').readlines()))
        new_links = [x.replace('\n', '') for x in new_links if
                     x.replace('\n', '') not in links]
        # Erase the new arxiv link file
        f = open(BASE+'update_arxiv_links.txt', 'w')
        f.close()
        # Update the favorite link file
        with open(BASE+'favorite_arxiv_links.txt', 'a') as f:
            for link in new_links:
                f.write(link + '\n')
        print('Score files updated with %d new links\n\n' % len(new_links),
              new_links, '\n', flush=True)
        # Update the score files
        author, title_abstract = self.load_score_files()
        for (idx, link) in enumerate(new_links):
            print(idx, end='  ', flush=True)
            a, t = self.generate_weight_from_url(link)
            for x in a:
                author[x] = author.get(x, 0.0) + a[x]
            for x in t:
                title_abstract[x] = title_abstract.get(x, 0.0) + t[x]
        with open(BASE+'author.pickle', 'wb') as f:
            pickle.dump(author, f)
        with open(BASE+'title_abstract.pickle', 'wb') as f:
            pickle.dump(title_abstract, f)
        # Print Top20 authors and keywords
        top20 = list(reversed(sorted(author, key=author.get)))[:20]
        print('\n\t\t === Top20 favoriate authors === \n')
        for x in top20:
            print(x.rjust(40), '%.3f' % author[x])
        top20 = list(reversed(sorted(
                        title_abstract, key=title_abstract.get)))[:20]
        print('\n\t\t === Top20 favoriate keywords === \n')
        for x in top20:
            print(x.rjust(40), '%.3f' % title_abstract[x])
        return self.unique_favorite_arxiv_links()

    def unique_favorite_arxiv_links(self, verbose=True):
        """ Unique and sort favorite arXiv links.

        Return the list of unique and sorted links
        """
        filename = BASE + 'favorite_arxiv_links.txt'
        links = list(set(open(filename).readlines()))
        links.sort()
        with open(filename, 'w') as f:
            for x in links:
                f.write(x)
        if verbose:
            print('\nFavorite arXiv links are unique and sorted.')
        return [x.replace('\n', '') for x in links]

    # def load_score_files(self):
    #     """ Load the score files """
    #     author = pickle.load(open(BASE+'author.pickle', 'rb'))
    #     title_abstract = pickle.load(open(BASE+'title_abstract.pickle', 'rb'))
    #     return (author, title_abstract)
    def load_score_files(self):
        """ Load the score files """
        author, title_abstract = dict(), dict()
        author_pickle_filepath = BASE+'author.pickle'
        if os.path.exists(author_pickle_filepath):
            if os.path.getsize(author_pickle_filepath) > 0:
                with open(author_pickle_filepath, 'rb') as f:
                    author = pickle.load(f)
        title_abstract_pickle_filepath = BASE+'title_abstract.pickle'
        if os.path.exists(title_abstract_pickle_filepath):
            if os.path.getsize(title_abstract_pickle_filepath) > 0:
                with open(title_abstract_pickle_filepath, 'rb') as f:
                    title_abstract = pickle.load(f)
        return (author, title_abstract)

    def get_content_from_url(self, url=EXAMPLE_URL):
        """ Get author, title, abstract from a url """
        content = {'author' : dict(), 'title' : '', 'abstract' : ''}
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.findAll("h1", {"class" : "title mathjax"})[0].get_text()
        author = soup.findAll("div", {"class" : "authors" })[0].get_text()
        abstract = soup.findAll("blockquote",
                                {"class" : "abstract mathjax"})[0].get_text()
        title, author, abstract = str(title), str(author), str(abstract)
        # author
        author = re.sub(r'\<[^>]*\>', '', author)  # rm content in '<...>'
        author = ''.join(author.split(':')[-1]).split('et al')[0]  # trim
        author = re.sub('\(.*?\)', '', author)  # rm affiliation
        author = author.replace('\n', ' ').split(',')
        author = [x.strip() for x in author]
        for x in author:
            content['author'][x] = content['author'].get(x, 0) + 1
        # title
        title = ''.join(title.split(':')[1:]).replace('\n', ' ')
        content['title'] = self.clean_text(title.lower())
        # abstract
        abstract = ''.join(abstract.split(':')[1:]).replace('\n', ' ')
        content['abstract'] = self.clean_text(abstract.lower())
        return content

    def score_from_url(self, url=EXAMPLE_URL, load_score=True,
                       score_files=(None, None), verbose=False):
        """ Given a url, return a score

        load_score controls if score_files are given or not
        """
        if load_score:
            author, title_abstract = self.load_score_files()
        else:
            author, title_abstract = score_files
        content = self.get_content_from_url(url)
        N = sum(content['author'].values()) * 1.0  # probably to divide; unused
        score1, score2, score3 = 0.0, 0.0, 0.0
        if verbose:
            print(content)
            print('\n\t\t=== AUTHOR SCORE : %d ===' % AUTHOR_SCORE)
        for x in content['author']:
            if x in author:
                if verbose:
                    print(x.rjust(50) + ' : ', author[x])
                score1 += content['author'][x] * author[x]
        if verbose:
            print('in total'.rjust(50) + ' : ', score1)
            print('\n\t\t=== TITLE SCORE : %d ===' % TITLE_SCORE)
        for x in title_abstract:
            if x in content['title']:
                if verbose:
                    print(x.rjust(50) + ' : ', title_abstract[x])
                score2 += title_abstract[x]
        if verbose:
            print('in total'.rjust(50) + ' : ', score2)
            print('\n\t\t=== ABSTRACT SCORE : %d ===' % ABSTRACT_SCORE)
        for x in title_abstract:
            if x in content['abstract']:
                if verbose and title_abstract[x] > 0.05 * score2:
                    print(x.rjust(50) + ' : ', title_abstract[x])
                score3 += title_abstract[x]
        score = (score1 * AUTHOR_SCORE + score2 * TITLE_SCORE
                 + score3 * ABSTRACT_SCORE)
        if verbose:
            print('...'.rjust(50) + ' : ' + '...')
            print('in total'.rjust(50) + ' : ', score3)
            print('\n\n' + 'TOTLE'.rjust(50) + ' : ', score)
        return score

    def score_from_entry(self, entry=dict(),
                         load_score=True, score_files=(None, None),
                         verbose=False):
        """ Given a feedparser entry, return a score """
        if load_score:
            author, title_abstract = self.load_score_files()
        else:
            author, title_abstract = score_files
        # Same as that in score_from_url
        title = entry['title'].split('arXiv:')
        status = title[-1].split(']')[-1][:-1].strip().lower()
        content = self.get_content_from_entry(entry)
        N = sum(content['author'].values()) * 1.0  # probably to divide; unused
        score1, score2, score3 = 0.0, 0.0, 0.0
        if verbose:
            print(content)
            print('\n\t\t=== AUTHOR SCORE : %d ===' % AUTHOR_SCORE)
        for x in content['author']:
            if x in author:
                if verbose:
                    print(x.rjust(50) + ' : ', author[x])
                score1 += content['author'][x] * author[x]
        if verbose:
            print('in total'.rjust(50) + ' : ', score1)
            print('\n\t\t=== TITLE SCORE : %d ===' % TITLE_SCORE)
        for x in title_abstract:
            if x in content['title']:
                if verbose:
                    print(x.rjust(50) + ' : ', title_abstract[x])
                score2 += title_abstract[x]
        if verbose:
            print('in total'.rjust(50) + ' : ', score2)
            print('\n\t\t=== ABSTRACT SCORE : %d ===' % ABSTRACT_SCORE)
        for x in title_abstract:
            if x in content['abstract']:
                if verbose and title_abstract[x] > 0.05 * score2:
                    print(x.rjust(50) + ' : ', title_abstract[x])
                score3 += title_abstract[x]
        score = (score1 * AUTHOR_SCORE + score2 * TITLE_SCORE
                 + score3 * ABSTRACT_SCORE)
        if len(status) > 0:  # update or cross-list
            score *= 0.5
        if verbose:
            print('...'.rjust(50) + ' : ' + '...')
            print('in total'.rjust(50) + ' : ', score3)
            print('\n\n' + 'TOTLE'.rjust(50) + ' : ', score)
        return score, [score1, score2, score3]


    def get_content_from_entry(self, entry=dict()):
        """ Get content from a feedparser entry """
        content = {'author' : dict(), 'title' : '', 'abstract' : ''}
        # title
        title = re.sub('\(.*?\)', '', entry['title'])[:-2].replace('\n', ' ')
        content['title'] = self.clean_text(title.lower())
        # abstract
        abstract = re.sub(r'\<[^>]*\>', '', entry['summary']).replace('\n', ' ')
        content['abstract'] = self.clean_text(abstract.lower())
        # author
        author = re.sub(r'\<[^>]*\>', '', entry['author'])  # rm '<...>'
        author = author.split('et al')[0]  # trim
        author = re.sub('\(.*?\)', '', author)  # rm affiliation
        author = author.replace('\n', ' ').split(',')
        author = [x.strip() for x in author]
        for x in author:
            content['author'][x] = content['author'].get(x, 0) + 1
        return content

    def generate_weight_from_url(self, url=EXAMPLE_URL):
        """ Generate score files from url

        Title and abstract are combined; author, title, abstract each with a
        total weight one.
        """
        author, title_abstract = dict(), dict()
        content = self.get_content_from_url(url)
        # author: every paper is normalised to 1
        N = sum(content['author'].values()) * 1.0
        for x in content['author']:
            author[x] = author.get(x, 0.0) + content['author'][x] / N
        # title
        r = Rake()
        r.extract_keywords_from_text(content['title'])
        keywords = r.get_ranked_phrases_with_scores()
        keywords = [x for x in keywords if len(''.join(x[1].split(' '))) >= 4
                    and len(x[1]) <=60]
        keywords = keywords[:int(len(keywords)/2)]
        tot = sum([x[0] for x in keywords])
        for (s, kw) in keywords:
            title_abstract[kw] = title_abstract.get(kw, 0.0) + 1.0 * s / tot
        # abstract
        r = Rake()
        r.extract_keywords_from_text(content['abstract'])
        keywords = r.get_ranked_phrases_with_scores()
        keywords = [x for x in keywords if len(''.join(x[1].split(' '))) >= 4
                    and len(x[1]) <=60]
        keywords = keywords[:int(len(keywords)/2)]
        tot = sum([x[0] for x in keywords])
        for (s, kw) in keywords:
            title_abstract[kw] = title_abstract.get(kw, 0.0) + 1.0 * s / tot
        return (author, title_abstract)

    def format_entry_to_html(self, entry):
        """ Format an entry from feedparser to html style """
        score, score_list = self.score_from_entry(entry, load_score=False,
                            score_files=(self.author, self.title_abstract))
        title = entry['title'].split('arXiv:')
        arxiv_id = title[-1].split('[')[0].strip().split('v')[0]
        status = title[-1].split(']')[-1][:-1].strip().lower()
        title = '  %d. ' % score + title[0][:-2]
        if len(status) > 0:
            title = title + ' [<font color="yellow">' + status + '</font>] '
        href = ('''[<a href="https://arxiv.org/abs/%s">arXiv</a>,
                <a href="https://arxiv.org/pdf/%s">PDF</a>,
                <a href="https://ui.adsabs.harvard.edu/#abs/arXiv:%s">ADS</a>]
                   ''' % (arxiv_id, arxiv_id, arxiv_id))
        scores = ('''  &#9632; author: %d &#9632; title: %d &#9632; abstract:
                  %d ''') % (int(score_list[0] * AUTHOR_SCORE),
                                 int(score_list[1] * TITLE_SCORE),
                                 int(score_list[2] * ABSTRACT_SCORE))
        abstract = re.sub(r'\</p\>', '', entry['summary'].replace('\n', ' '))
        abstract = re.sub(r'\<p\>', '', abstract)
        abstract = '''<p style="margin-left: 40px; 
                         margin-right: 50px; 
                         width: 1000px; 
                         border:3px; 
                         border-style:solid; 
                         border-color:#454545; 
                         padding: 1em; 
                         background-color:rgb(240, 240, 240)" 
                         align="left" 
                         class="big">
                         <font face="arial">''' + entry['author'] + "<br><br>" + abstract + "<br><br>" + href + scores + '</font></p>'  # font
        html = '\n<h2 style="margin-left: 40px; margin-right: 50px; background-color:DodgerBlue; display:inline-block; padding:1em; width: 1000px; color:white">' + title + '</h2>' + abstract + '\n'
        return (score, html)

    def run_daily_arXiv(self):
        """ Run daily arXiv feed """
        links, entries = [], []
        # Prepare entries with scores
        for rss_url in RSS_URLS:
            print(' ... parse ', rss_url)
            feed = feedparser.parse(rss_url)
            for entry in feed["entries"]:
                link = entry['link']
                if link not in links:
                    links.append(link)
                    entries.append(self.format_entry_to_html(entry))
        zz = sorted([x[0] for x in entries])
        if len(zz) >= 5:
            thres = (zz[-3]*zz[-4]*zz[-5])**(1.0/3) / 3.0 # only keep high scores
        else:
            thres = 0.0
        papers = dict()
        for (s, e) in entries:
            if s > thres:
                papers[s] = e
        idx = list(reversed(sorted(papers.keys())))

        print('\nIn total: %d entries\n' % len(entries))
        # Write to html file
        count = 0
        browser_cmd = "open"
        out_file_name = BASE + \
                time.strftime("%Y%m%d") + ".html"
        f = open(out_file_name, "w");
        f.write('<html xmlns="http://www.w3.org/1999/xhtml" lang="en"> \n')
        f.write('<head><title>arXiv daily</title> \n' + MATHJAX + '</head>\n')
        f.write('<body class="with-cu-identity"> \n')
        f.write('<div id="footer"></div> \n')
        f.write('<center><h1 style="margin-left: 40px; margin-right: 50px; background-color:SlateBlue; display:inline-block; padding:1em; width: 1000px; color:yellow">arXiv:' + time.strftime("%Y-%m-%d") +
                '</h1>')
        f.write('<div id="footer"></div> \n')
        for s in idx:
            f.write(papers[s])
            count += 1
        f.write('</center><div id="footer"></div> \n')
        f.write('</body></html> \n')
        f.close()
        os.system(browser_cmd + " " + out_file_name)
        print('''\nIn total %d articles for your preview; please enjoy and have
              a good day!\n''' % count)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(sys.argv[0], sys.argv[1])
        my_arxiv = arXiv(sys.argv[1])
    else:
        my_arxiv = arXiv()
        my_arxiv.run_daily_arXiv()

