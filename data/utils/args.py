import os


def is_existing_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(f'File {arg} does not exist!')
    else:
        return arg


def is_valid_folder(parser, arg):
    try:
        os.makedirs(os.path.dirname(arg), exist_ok=True)
        return arg
    except Exception:
        parser.error(f'Path {arg} is not a valid path!')
