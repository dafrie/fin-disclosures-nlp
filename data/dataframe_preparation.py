from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import WordNetLemmatizer
import os
from pathlib import Path
import re
import string

from tqdm.auto import tqdm
import yaml
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

import spacy
from spacy.lang.en import English

from data.preprocessing import DocumentPreprocessor
import data

nlp = spacy.load('en_core_web_md', disable=[
                 "parser", "ner", "entity_linker", "textcat", "entity_ruler"])

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


def get_count_matrix(doc, vocabulary):
    count_vectorizer = CountVectorizer(ngram_range=(
        1, 2), vocabulary=vocabulary, tokenizer=spacy_tokenizer)
    count_matrix = count_vectorizer.fit_transform(doc)
    count_df = pd.DataFrame(count_matrix.toarray(
    ), columns=count_vectorizer.get_feature_names())
    return count_df


def get_text_from_page(path, page_no):
    if os.path.isfile(path):
        with open(path, 'r') as stream:
            content = yaml.safe_load(stream)
            page = content['pages'][page_no - 1]
            return page['text']


def get_document(path):
    if os.path.isfile(path):
        with open(path, 'r') as stream:
            content = yaml.safe_load(stream)
            return content['pages']


def get_counts_per_page(path, vocabulary):
    texts = []
    if os.path.isfile(path):
        with open(path, 'r') as stream:
            try:
                content = yaml.safe_load(stream)
                for page in content['pages']:
                    text = DocumentPreprocessor(
                        page['text']).fix_linebreaks()
                    texts.append(text)
                count_vectorizer = CountVectorizer(ngram_range=(
                    1, 2), vocabulary=vocabulary, tokenizer=spacy_tokenizer)
                count_matrix = count_vectorizer.fit_transform(texts)
                count_df = pd.DataFrame(count_matrix.toarray(
                ), columns=count_vectorizer.get_feature_names())
            except yaml.YAMLError as exc:
                print(exc)
        count_df = count_df[count_df.sum(axis=1) > 0]
        count_df = count_df.loc[:, (count_df != 0).any(axis=0)]
        count_df.index += 1  # Shift index by one so it aligns with actual page numbers
        count_df.index.name = "page_no"
        return count_df


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


def get_toc_from_yaml(path):
    if os.path.isfile(path):
        with open(path, 'r') as stream:
            try:
                content = yaml.safe_load(stream)
                if content['toc'] and len(content['toc']):
                    return content['toc']
            except yaml.YAMLError as exc:
                print(exc)
            return


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

    # Spacy is quite memory hungry, thus split the doc...
    max_length = 10000
    words = text.split()
    split_docs = [' '.join(words[i:i+max_length])
                  for i in range(0, len(words), max_length)]

    def _tokenize(text):
        # Creating our token object, which is used to create documents with linguistic annotations.
        _tokens = nlp(text)

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


def get_company_paths(path):
    result = []
    for f in os.scandir(path):
        if not f.name.startswith('.') and f.is_dir():
            result.append(f)
    return result


def get_reports_paths(path):
    result = []
    for f in os.scandir(path):
        p = Path(f.path)
        if not f.name.startswith('.') and p.suffix == '.pdf' and f.is_file():
            result.append(p)
    return result


def get_df(input_path='../input_files/files', report_type_mappings={}, selected_report_types={}, include_text=True, include_page_no=True, include_toc=True):
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

    # Define dataframe columns
    column_names = ['id', 'country', 'company', 'orig_report_type',
                    'report_type', 'year', 'input_file', 'output_file']
    if include_toc:
        column_names.append('toc')
    if include_text:
        column_names.append('text')

    company_paths = get_company_paths(input_path)
    company_loop = tqdm(company_paths)
    company_loop.set_description(f"Overall progress")

    # Loop through companies
    for company_path in company_loop:
        c = company_path.name.partition('_')
        if len(c[2]) > 0:
            country = c[0] if c[0] else np.NaN
            company = c[2] if c[2] else np.NaN
        else:
            country = "NotAvailable"
            company = c[0]

        # Loop through files within
        company_files = get_reports_paths(company_path.path)
        files_loop = tqdm(company_files, leave=False)
        files_loop.refresh()
        company_loop.update()
        files_loop.set_description(f"{company}")
        for p in files_loop:
            path = os.path.relpath(p, p.parent.parent)
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
                    f"Invalid document found at {p} with year {year}")
                continue
            report_id = f"{country}_{company}-{orig_report_type}_{year}"

            # Check if output folder and file exists
            expected_output_path = os.path.join(
                company_path.path, p.stem, p.stem + '.yml')
            o = Path(expected_output_path)
            output_file = o.name if os.path.isfile(
                expected_output_path) else np.NaN

            # Mandatory fields
            row = [report_id, country, company, orig_report_type,
                   report_type, year, path, output_file]

            if include_toc:
                toc = get_toc_from_yaml(expected_output_path)
                toc = toc if toc else np.NaN
                row.append(toc)

            if include_text:
                text = get_text_from_yaml(
                    expected_output_path, include_page_no=include_page_no)
                text = text if text else np.NaN
                row.append(text)

            input_files.append(row)
            files_loop.update()

    df = pd.DataFrame(input_files, columns=column_names)
    # Filter out report types that are not selected
    if len(selected_report_types):
        df = df[df['report_type'].isin(selected_report_types)]
    return df


def filter_line(line):
    if not len(line) or line.startswith('#'):
        return None
    return line


def get_keywords_from_file(file_name, should_lemmatize=True):
    with open(os.path.join(file_name)) as f:
        keywords = [r for r in (filter_line(line.strip())
                                for line in f.readlines()) if r is not None]
        if should_lemmatize:
            result = []
            for entry in keywords:
                tokens = nlp(entry)
                tokens = [token.lemma_.lower().strip() if token.lemma_ !=
                          "-PRON-" else token.lower_ for token in tokens]
                result.append(" ".join(tokens))
            keywords = result
    return keywords
