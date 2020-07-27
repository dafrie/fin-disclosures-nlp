import sys
import os
import shutil
from io import StringIO
import yaml
import tempfile
import logging
import argparse
from pathlib import Path

import tika
from bs4 import BeautifulSoup
import pytesseract
from tqdm import tqdm

from tika import parser
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines
from pdfminer.pdfparser import PDFParser
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.psparser import PSKeyword, PSLiteral, LIT
from pdfminer.pdftypes import PDFStream, PDFObjRef, resolve1, stream_value
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.high_level import extract_text, extract_text_to_fp
from nltk.tokenize import RegexpTokenizer
import spacy
from spacy.matcher import PhraseMatcher

import pandas as pd

from utils.timer import timer
from utils.args import is_existing_file, is_valid_folder

try:
    from PIL import Image
except ImportError:
    import Image


input_file = "./data/samples/test/UBS_TEST.pdf"
output_folder = "./data/samples/test/SR_2019/"
similarity_threshold = 0.4

# TODO: Takes long to instantiate, so should be done just per thread!
nlp = spacy.load("en_core_web_md")

matcher = PhraseMatcher(nlp.vocab)
terms = ["climate change", "greenhouse gas",
         "global warming", "emissions", "co2", "renewables"]
# Only run nlp.make_doc to speed things up
patterns = [nlp.make_doc(text) for text in terms]
matcher.add("TerminologyList", None, *patterns)

# Initialize tokens
# TODO: How can we use Phrase matcher so we can check tokens?
topic_tokens = nlp(
    "climate emissions co2 carbon greenhouse ghg renewables footprint")


def get_max_similarity(word):
    if not word.has_vector:
        return 0
    results = []
    for token in topic_tokens:
        sim = token.similarity(word)
        results.append(sim)
    return max(results)


class PdfExtractor:
    def __init__(self, input_file, output_folder=None, parser="tika", ocr_strategy="no_ocr", **kwargs):
        self.logger = logging.getLogger('pdf_extractor')
        self.input_file = input_file
        self.output_folder = output_folder
        self.parser = parser
        self.ocr_strategy = ocr_strategy
        self.process_document()

    def extract_toc(self):
        """ Returns the extracted table of contents of the document (if found)

        Credits
        -------
        https://github.com/pdfminer/pdfminer.six/blob/master/tools/dumppdf.py

        Returns
        -------
        list: list of dict of { str: int, str: str, str: int}
            A list of dictionaries containing the "level", "title" and "page_no"
        """
        toc = []
        with open(self.input_file, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            pages = {page.pageid: pageno for (pageno, page)
                     in enumerate(PDFPage.create_pages(doc), 1)}

            def resolve_dest(dest):
                if isinstance(dest, str):
                    dest = resolve1(doc.get_dest(dest))
                elif isinstance(dest, PSLiteral):
                    dest = resolve1(doc.get_dest(dest.name))
                if isinstance(dest, dict):
                    dest = dest['D']
                if isinstance(dest, PDFObjRef):
                    dest = dest.resolve()
                return dest

            try:
                outlines = doc.get_outlines()
                # For each found outline/bookmark, resolve the page number of the object
                for (level, title, dest, a, se) in outlines:
                    page_no = None
                    if dest:
                        dest = resolve_dest(dest)
                        page_no = pages[dest[0].objid]
                    elif a:
                        action = a.resolve()
                        if isinstance(action, dict):
                            subtype = action.get('S')
                            if subtype and repr(subtype) == '/\'GoTo\'' and action.get('D'):
                                dest = resolve_dest(action['D'])
                                page_no = pages[dest[0].objid]
                    toc.append(
                        {"level": level, "title": title, "page_no":  page_no})
            except PDFNoOutlines:
                print("No outline for PDF found!")
                pass
            except Exception:
                print("General error getting outline for PDF")
            parser.close()
        return toc

    def extract_with_tika(self):
        """
        Note that pytika can be additionally configured via environment variables in the docker-compose file!
        """
        pages_text = []

        # Read PDF file and export to XML to keep page information
        data = parser.from_file(
            str(self.input_file),
            xmlContent=True,
            requestOptions={'timeout': 6000},
            # 'X-Tika-PDFextractInlineImages' : true # Unfortunately does not really work
            # Options: 'no_ocr', 'ocr_only' and 'ocr_and_text'
            headers={'X-Tika-PDFOcrStrategy': self.ocr_strategy}
        )
        xhtml_data = BeautifulSoup(data['content'], features="lxml")

        pages = xhtml_data.find_all('div', attrs={'class': 'page'})
        for i, content in enumerate(tqdm(pages, disable=False)):
            _buffer = StringIO()
            _buffer.write(str(content))
            parsed_content = parser.from_buffer(_buffer.getvalue())

            text = ''
            if parsed_content['content']:
                text = parsed_content['content'].strip()
            excertp = text.replace('\n', ' ')
            pages_text.append({"page_no": i+1, "text": text})
        return pages_text

    def process_images(self, tmp_dir):
        """ Runs tesseract OCR on images found in the specified folder. Calculates max similarity for each word to set of initial words
        """
        images = os.listdir(tmp_dir)
        full_text = ''
        unprocessed_images = []
        relevant_images = []
        for i in images:
            try:
                text = pytesseract.image_to_string(
                    Image.open(os.path.join(tmp_dir, i))) + '\n\n'
                full_text += text
                ocr_tokens = nlp(text)
                df = pd.DataFrame(ocr_tokens, columns=['ocr_token'])
                df['result'] = df['ocr_token'].apply(
                    lambda x: get_max_similarity(x))
                if df['result'].max() > similarity_threshold:
                    relevant_images.append(i)

            except Exception as error:
                self.logger.warning(
                    f'Exception processing image! {i} Message: {error}')
                unprocessed_images.append(i)

        return full_text, unprocessed_images, relevant_images

    def extract_with_pdfminer(self):
        pages_text = []

        with open(self.input_file, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            pages = {page.pageid: pageno for (pageno, page)
                     in enumerate(PDFPage.create_pages(doc), 1)}
            for idx, page in enumerate(tqdm(pages, disable=False)):
                _buffer = StringIO()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # If output_dir is given the extracted images are stored there.
                    extract_text_to_fp(
                        fp, outfp=_buffer, page_numbers=[idx], output_dir=tmp_dir)
                    text_from_images, unprocessed_images, relevant_images = self.process_images(
                        tmp_dir)
                    text = _buffer.getvalue() + '\n\n' + text_from_images

                    if len(unprocessed_images):
                        self.logger.info(
                            f'Ignoring {len(unprocessed_images)} unprocessable image(s)....')

                    # Move relevant files to save place
                    for i in relevant_images:
                        path = os.path.join(
                            self.output_folder, 'relevant_images')
                        os.makedirs(path, exist_ok=True)
                        shutil.move(os.path.join(tmp_dir, i),
                                    os.path.join(path, i))

                    pages_text.append(
                        {"page_no": idx + 1, "text": text, "unprocessed_images": unprocessed_images, "relevant_images": relevant_images})
        return pages_text

    @timer
    def extract_text_per_page(self):
        if self.parser == "tika":
            return self.extract_with_tika()
        return self.extract_with_pdfminer()

    def write_output(self):
        output = {
            "toc": self.toc,
            "pages": self.pages_text
        }
        filename = Path(self.input_file).stem
        out_file_path = os.path.join(self.output_folder, filename)
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        with open(out_file_path + '.yml', 'w') as fp:
            yaml.dump(output, fp)
        """" 
        with open(out_file_path + '.txt', 'w') as fp:
            text = '\n\n<pagebreak />\n\n'.join(output['pages']['text'])
            fp.write(text)
        """

    def process_document(self):
        try:
            self.toc = self.extract_toc()
        except Exception as error:
            self.toc = []
            self.logger.error(
                f'Exception getting TOC! Message: {error}')
        self.pages_text = self.extract_text_per_page()
        self.write_output()


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument(
        'input_file', help="Path to the input file", default=input_file, metavar="FILE", type=lambda f: is_existing_file(parser, f))
    parser.add_argument(
        'output_folder', help="Path to the output folder", default=output_folder, metavar="FOLDER", type=lambda f: is_valid_folder(parser, f))

    # Optional arguments
    parser.add_argument("-d", "--debug", help="Debug",
                        type=bool, default=False)
    parser.add_argument("-l", "--log-level", help="Log level",
                        type=str, default="info")
    parser.add_argument('-f', "--log-file",
                        help="Log file location. By default, it stores the log file in the output directory", default=None)
    parser.add_argument("-p", "--parser", help="Specify the PDF parser",
                        choices=('tika', 'pdfminer'), default="pdfminer")
    parser.add_argument("-o", "--ocr-strategy", help="Specify the OCR Strategy",
                        choices=('no_ocr', 'ocr_only', 'ocr_and_text'), default="no_ocr")
    args = parser.parse_args()
    return args


def main(**kwargs):
    log_file = kwargs['log_file'] if kwargs.get('log_file') else None
    log_level = logging.info if not kwargs.get('debug') else logging.debug
    log_level = getattr(logging, kwargs['log_level'].upper()) if kwargs.get(
        'log_level') else log_level

    logging.basicConfig(filename=log_file)
    logger = logging.getLogger('pdf_extractor')
    logger.setLevel(log_level)
    logger.info(f'PDF Extractor CLI initialized with arguments: \n{kwargs}')

    # Run actual extractor
    PdfExtractor(**kwargs)


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
