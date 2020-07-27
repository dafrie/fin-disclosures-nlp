"""A PDF Crawler of different types of disclosure documents from company websites

    How to use:
    - Update the config below
    - Run "python pdf_crawler.py"
    - Check the validity of the results, select accordingly
"""


import os
import requests
import pprint
import json
import itertools
import tempfile
import subprocess

import tqdm
from PyInquirer import style_from_dict, Token, prompt, print_json

from collections import OrderedDict
from requests_html import HTMLSession, AsyncHTMLSession
from urllib.parse import urljoin, urlparse, urlunsplit
from bs4 import BeautifulSoup
from fuzzywuzzy import process, fuzz


pp = pprint.PrettyPrinter(indent=4, )

# ================================ CONFIG ==========================================
company = "nl_ing_grp"
url = "https://www.ing.com/About-us/Annual-reporting-suite/Annual-Reports-archive.htm"
start_year = 1999
end_year = 2019
document_type = 'AR'
api = False
dynamic = True
identifier = "text"

directory = "../stoxxEurope50"
user_agent = 'Mozilla/5.0 (Linux; Android 6.0.1; SM-G935S Build/MMB29K; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/55.0.2883.91 Mobile Safari/537.36'
headers = {'User-Agent': user_agent}


path = os.path.join(directory, company)

# Config for keywords
config = {
    "AR": {
        "tags": {
            "positive": ['Annual', 'Groep'],
        }
    },
    "20F": {
        "tags": {
            "positive": ["20-F"]
        }
    },
    "SR": {
        "tags": {
            "positive": ['Environmental']
        }
    }
}
# ====================================================================================


def extract_pdf_links(urls):
    """Filters the input list of urls to look for PDF files. Also removes the rest of the path after the filename

    Parameters
    ----------
    urls : [string]
        A list of urls

    Returns
    -------
    [string]
        A filtered list of input urls
    """
    pdfs = []
    for url in urls:
        u, sep, tail = url.lower().partition('.pdf')
        url = u + sep
        if ".pdf" in url or ".PDF" in url:
            pdfs.append(url)
    return pdfs


def find_keys(node, key):
    """Traverses recursivly a nested list/dictionary (usually from a JSON result) and collects all named keys

    Parameters
    ----------
    node : dict, list
        Nested list or dictionary that the key should get extracted from
    key : string
        The named key that should get extracted

    Yields
    -------
    dict, list
        Returns the list of found named keys and their values
    """
    if isinstance(node, list):
        for i in node:
            for x in find_keys(i, key):
                yield x
    elif isinstance(node, dict):
        if key in node:
            yield node[key]
        for j in node.values():
            for x in find_keys(j, key):
                yield x


def getValueFromElement(el):
    """Quick helper to extract relevant content from a HTML element

    Parameters
    ----------
    el : Element
        A requests-html element type

    Returns
    -------
    string
        The relevant value of the element
    """
    if identifier == "link":
        return list(el.absolute_links)[0]
    return el.text


class PdfCrawler:
    """Base class for getting URLs from a website, calculate the best match for each year and then download asynchronous
    """
    asession = AsyncHTMLSession()

    def __init__(self, document_type, start_year, end_year, url):
        self.document_type = document_type
        self.start_year = start_year
        self.end_year = end_year
        self.url = url
        self.document_links = OrderedDict()

    def get_links(self):
        return []

    def get_best_match(self, year, links):
        return False

    def calculate_best_matches(self, links):
        result = OrderedDict()
        for year in range(self.start_year, self.end_year + 1):
            match = self.get_best_match(year, links)
            if match:
                result.update({year: match})
        self.document_links = result

    def get_full_url(self, url):
        o = urlparse(url)
        if o.scheme != '':
            return url
        else:
            url_o = urlparse(self.url)
            return urlunsplit([url_o.scheme, url_o.netloc, o.path, '', ''])

    def prepare_links(self):
        links = self.get_links()
        self.calculate_best_matches(links)

    def preview_links(self):
        self.prepare_links()
        print("============= RESULTS =============")
        print(json.dumps(self.document_links, indent=1))

    async def fetch(self, url, filename, unlock=True):
        r = await self.asession.get(url, headers=headers)
        with open(filename, 'wb') as f:
            f.write(r.content)
        # Many PDFs have an empty password and can not be modified
        # Note: Requires qpdf installed!
        if unlock:
            subprocess.run(["qpdf", "--decrypt --replace-input", filename])

    def download_links(self):
        self.prepare_links()
        questions = [
            {
                'type': 'confirm',
                'name': 'confirm_list',
                'message': "Should selection be skipped and download all at once?"
            },
            {
                'type': 'checkbox',
                'qmark': '--->',
                'message': 'Select links',
                'name': 'document_links',
                'choices': [{'name': str(key) + ': ' + val, 'checked': False} for key, val in self.document_links.items()],
                'when': lambda answers: not answers['confirm_list']
            }]
        answers = prompt(questions)
        if not answers["confirm_list"]:
            links = OrderedDict()
            for choice in answers["document_links"]:
                links.update({int(choice[0:4]): choice[6:]})
            self.document_links = links

        no_documents = len(self.document_links)
        if no_documents > 0:
            tasks = [lambda url=url, year=year: self.fetch(url, os.path.join(path, f"{self.document_type}_{year}.pdf"))
                     for year, url in self.document_links.items()]
            print(
                f"Starting to download {no_documents} PDF's async into {path}")
            self.asession.run(*tasks)
            print(f"Download completed!")
        else:
            print("No files needed to download...")


class DynamicPdfCrawler(PdfCrawler):
    """Extension to the PDF crawler to get PDF urls from website's static HTML and HTML rendered by JavaScript
    """

    def get_links(self):
        session = HTMLSession()
        r = session.get(self.url, headers=headers)
        r.html

        # Run JavaScript code on webpage
        if dynamic:
            r.html.render()
        anchors = r.html.find('a')
        links = {}
        for idx, a in enumerate(anchors):
            if len(a.absolute_links) > 0:
                # Should only be ever max one
                # filenames = [os.path.basename(
                #     urlparse(link).path) for link in a.absolute_links]
                pdfs = extract_pdf_links(a.absolute_links)
                if len(pdfs):
                    links.update({idx: a})
        return links

    def get_best_match(self, year, links):
        texts = {key: getValueFromElement(val) for key, val in links.items()}
        # First filter by year:
        filtered = {idx: l for l, score, idx in process.extract(
            str(year), texts, scorer=fuzz.partial_ratio) if score > 85}
        # Then get the best match with the positive keywords
        best_match = process.extractOne(
            ' '.join(config[document_type]["tags"]["positive"]), filtered, scorer=fuzz.partial_token_sort_ratio)
        if best_match:
            score = best_match[1]
            best_match = links[best_match[2]]
            link = list(best_match.absolute_links)[0]
            path = urlparse(link)
            path = os.path.basename(path.path)
            print(year, "| Best match: ", best_match.text[-60:] +
                  ' - ' + link[-60:], ' Score: ', score)
            return link
        else:
            print(year, "| No match found!")
            return None


class ApiPdfCrawler(PdfCrawler):
    def get_links(self):
        session = HTMLSession()
        r = session.get(self.url, headers=headers)
        json = r.json()
        links = list(find_keys(json, 'basicUrl'))
        links = list(map(lambda l: self.get_full_url(l), links))
        links = extract_pdf_links(links)
        return links

    def get_best_match(self, year, links):
        # First filter by year:
        filtered = [l for l, score in process.extract(
            str(year), links, scorer=fuzz.partial_ratio) if score > 80]
        """
            best_matches = process.extract(
                str(year) + ' ' + ' '.join(config[document_type]["tags"]["positive"]), filtered, scorer=fuzz.partial_token_sort_ratio
            )
            # TODO(df): Also filter for short hand YY!
            # TODO(df): Exclude reports that have a YYYY/YY in the filename but also one in the path!
            """
        best_match = process.extractOne(
            str(year) + ' ' + ' '.join(config[document_type]["tags"]["positive"]), filtered)
        if best_match:
            path = urlparse(best_match[0])
            path = os.path.basename(path.path)
            print(year, "| Best match: ", path, ' Score: ', best_match[1])
        else:
            print(year, "| No match found!")
        val = best_match[0] if best_match else None
        return val


if __name__ == '__main__':
    if api:
        crawler = ApiPdfCrawler(document_type=document_type,
                                start_year=start_year, end_year=end_year, url=url)
    else:
        crawler = DynamicPdfCrawler(document_type=document_type,
                                    start_year=start_year, end_year=end_year, url=url)

    crawler.download_links()
