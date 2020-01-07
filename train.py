from argparse import ArgumentParser
from pprint import pprint

from duet_utils.io import load_json_file

if __name__ == '__main__':
    parser = ArgumentParser(description='Train the DUET model.')
    parser.add_argument('CONF_FILE', type=str, help='Path to the training config json file.')
    args = parser.parse_args()

    conf = load_json_file(args.CONF_FILE)

    pprint(conf)
