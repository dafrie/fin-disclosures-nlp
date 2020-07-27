import os
from pathlib import Path
import re

import yaml
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

try:
    from nltk import wordnet
except ImportError:
    nltk.download('wordnet')
from nltk import WordNetLemmatizer

try:
    from nltk import punkt
except ImportError:
    nltk.download('punkt')

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


def get_text_from_yaml(path):
    text = ''
    if os.path.isfile(path):
        with open(path, 'r') as stream:
            try:
                content = yaml.safe_load(stream)
                for page in content['pages']:
                    text += page['text']
            except yaml.YAMLError as exc:
                print(exc)
    return text


def tokenize(text):
    """

    """
    tokens = nltk.word_tokenize(text)
    words = []
    wnl = WordNetLemmatizer()
    for t in tokens:
        words.append(wnl.lemmatize(word=t))
    return words


def get_df(input_path='../files', report_type_mappings={}, selected_report_types={}):
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

                    text = get_text_from_yaml(expected_output_path)
                    text = text if text else np.NaN

                    input_files.append(
                        [country, company, orig_report_type, report_type, year, entry.name, output_file, text])
    df = pd.DataFrame(input_files, columns=[
                      'country', 'company', 'orig_report_type', 'report_type', 'year', 'input_file', 'output_file', 'text'])
    # Filter out report types that are not selected
    if len(selected_report_types):
        df = df[df['report_type'].isin(selected_report_types)]
    return df
