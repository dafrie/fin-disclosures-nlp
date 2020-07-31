from nltk import WordNetLemmatizer
import os
from pathlib import Path
import re
import string

import yaml
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

import spacy
from spacy.lang.en import English
nlp = spacy.load('en_core_web_md')

try:
    from nltk import wordnet
except ImportError:
    nltk.download('wordnet')

try:
    from nltk import punkt
except ImportError:
    nltk.download('punkt')

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


def get_text_from_yaml(path, include_page_no=True):
    text = ''
    if os.path.isfile(path):
        with open(path, 'r') as stream:
            try:
                content = yaml.safe_load(stream)
                for idx, page in enumerate(content['pages']):
                    if include_page_no:
                        text += '======= Page: ' + \
                            str(idx + 1) + ' =======\n\n\n'
                    text += page['text'] + '\n\n\n'
            except yaml.YAMLError as exc:
                print(exc)
    return text


def tokenize(text):
    """

    """
    stop_words = stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    words = []
    wnl = WordNetLemmatizer()
    for t in tokens:
        if not t in stop_words:
            words.append(wnl.lemmatize(word=t))
    return words


def spacy_tokenizer(text):
    """Tokenizes, removes stop words and lemmatizes the input text using Spacy NLP

    Parameters
    ----------
    text : String
        Input text to process

    Returns
    -------
    String[]
        A list of tokens
    """
    # Setup spacy
    punctuations = string.punctuation
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    parser = English()

    # Spacy is quite memory hungry, thus split the doc...
    max_length = 1000000
    split_docs = [text[index: index + max_length]
                  for index in range(0, len(text), max_length)]

    def _tokenize(text):
        # Creating our token object, which is used to create documents with linguistic annotations.
        _tokens = parser(text)

        # Lemmatizing each token and converting each token into lowercase
        _tokens = [word.lemma_.lower().strip() if word.lemma_ !=
                   "-PRON-" else word.lower_ for word in _tokens]

        # Removing stop words
        _tokens = [
            word for word in _tokens if word not in stop_words and word not in punctuations]

        # return preprocessed list of tokens
        return _tokens

    tokens = []
    for d in split_docs:
        tokens.extend(_tokenize(d))
    return tokens


def get_df(input_path='../input_files/files', report_type_mappings={}, selected_report_types={}, include_text=True, include_page_no=True):
    """Collects and returns a dataframe containing all reports by type and year found in each companies folder

    Parameters
    ----------
    input_path : str, optional
        Input path containing folders of companies with their reports in PDF form, by default '../files'
    report_type_mappings : dict, optional
        An optional dictionary to map report types to another, e.g. CSR to SR --> {"CSR": "SR"}. By default {}

    Returns
    -------
    Pandas Dataframe
        A dataframe
    """
    input_files = []
    # Loop through companies
    for company_path in os.scandir(input_path):
        if not company_path.name.startswith('.') and company_path.is_dir():
            c = company_path.name.partition('_')
            country = c[0] if c[0] else np.NaN
            company = c[2] if c[2] else np.NaN
            # Loop through files within
            for entry in os.scandir(company_path.path):
                p = Path(entry.path)
                if not entry.name.startswith('.') and p.suffix == '.pdf' and entry.is_file():
                    r = p.stem.split('_')
                    orig_report_type = r[0] if len(r) > 0 else np.NaN
                    report_type = report_type_mappings[
                        orig_report_type] if orig_report_type in report_type_mappings else orig_report_type
                    year = r[1] if len(r) > 1 else np.NaN

                    # Skip if there is no valid year...
                    try:
                        year = int(year)
                    except ValueError:
                        print(
                            f"Invalid document found at {entry.path} with year {year}")
                        continue

                    # Check if output folder and file exists
                    expected_output_path = os.path.join(
                        company_path.path, p.stem, p.stem + '.yml')
                    o = Path(expected_output_path)
                    output_file = o.name if os.path.isfile(
                        expected_output_path) else np.NaN

                    row = [country, company, orig_report_type,
                           report_type, year, entry.name, output_file]

                    if include_text:
                        text = get_text_from_yaml(
                            expected_output_path, include_page_no=include_page_no)
                        text = text if text else np.NaN
                        row.append(text)

                    input_files.append(row)
    df = pd.DataFrame(input_files, columns=[
                      'country', 'company', 'orig_report_type', 'report_type', 'year', 'input_file', 'output_file', 'text'])
    # Filter out report types that are not selected
    if len(selected_report_types):
        df = df[df['report_type'].isin(selected_report_types)]
    return df
