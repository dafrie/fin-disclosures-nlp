import sys
import os
import logging
import argparse
from datetime import datetime
from pathlib import Path, PurePath
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from pdf_extractor import PdfExtractor


def process_file(input_file, output_folder, **kwargs):
    print(f'Processing file {input_file} into {output_folder}')
    PdfExtractor(input_file=input_file, output_folder=output_folder, **kwargs)
    print(f'Done processing file {input_file}')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_master_file_path', help="File path of the input master file", metavar="PATH")

    parser.add_argument(
        'output_path', help="Folder path of the output directory", metavar="FOLDER")

    args = parser.parse_args()
    return args


def main(**kwargs):
    input_master_file_path = kwargs.get('input_master_file_path')
    output_path = kwargs.get('output_path')

    df = pd.read_csv(input_master_file_path)

    executor = ProcessPoolExecutor(max_workers=8)
    futures = []

    for (idx, row) in df.iterrows():
        if not row.extracted:
            futures.append(executor.submit(
                process_file, Path(row.path), PurePath(output_path, row.company)))

    for future in futures:
        future.result()


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
