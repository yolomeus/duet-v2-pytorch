import json
import pickle


def load_json_file(filepath):
    """Load a python dict with the contents of a json file.

    Args:
        filepath (str): path to the json file to load.

    Returns:
        dict: dictionary corresponding to the json file.

    """
    with open(filepath, 'r') as json_fp:
        return json.load(json_fp)


def load_pkl_file(filepath):
    """Load the contents of a pickle file.

        Args:
            filepath (str): path to the pickle file to load.

        Returns:
            object: the object stored in the pickle file.

        """
    with open(filepath, 'rb') as pkl_fp:
        return pickle.load(pkl_fp)
