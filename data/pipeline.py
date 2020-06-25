import sys
import os
import logging
import argparse
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

from pdf_extractor import PdfExtractor
from utils.args import is_existing_file, is_valid_folder


def get_files(input_folder):
    """Assume that the input_folder contains multiple folders containing files (so we look 1 level deep!)
    """
    files = []
    for company in os.scandir(input_folder):
        if not company.name.startswith('.') and company.is_dir():
            for entry in os.scandir(company.path):
                if not entry.name.startswith('.') and entry.is_file():
                    files.append(os.path.abspath(entry.path))
    return files


def process_file(file, kwargs):
    parser = kwargs['parser']
    file = Path(file)
    filename = file.stem
    output_folder = os.path.join(file.parent, filename)
    print(f'Processing file {file} into {output_folder} with parser {parser}')
    PdfExtractor(input_file=file, output_folder=output_folder,
                 parser=parser)
    print(f'Done processing file {file}')


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument(
        'input_folder', help="Path of the input folder containing company folders", default='./data/samples', metavar="FOLDER", type=lambda f: is_valid_folder(parser, f))

    # Optional arguments
    parser.add_argument("-d", "--debug", help="Debug",
                        type=bool, default=False)
    parser.add_argument("-l", "--log-level", help="Log level",
                        type=str, default="info")

    parser.add_argument("-p", "--parser", help="Specify the PDF parser",
                        choices=('tika', 'pdfminer'), default="tika")
    args = parser.parse_args()
    return args


def main(**kwargs):
    input_folder = kwargs.get('input_folder')
    log_file = kwargs['log_file'] if kwargs.get('log_file') else os.path.join(
        input_folder, f'log_{datetime.today().isoformat()}.txt')
    log_level = logging.info if not kwargs.get('debug') else logging.debug
    log_level = getattr(logging, kwargs['log_level'].upper()) if kwargs.get(
        'log_level') else log_level

    logging.basicConfig(filename=log_file)
    logger = logging.getLogger('pipeline')
    logger.setLevel(log_level)
    logger.info(f'Pipeline initialized with arguments: \n{kwargs}')

    executor = ProcessPoolExecutor(max_workers=8)
    files = get_files(kwargs.get('input_folder'))

    futures = []
    for f in files:
        futures.append(executor.submit(process_file, f, kwargs))

    for future in futures:
        future.result()


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
