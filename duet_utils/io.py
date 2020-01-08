import json
import pickle
from importlib import import_module


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


def dump_pkl_file(obj, filepath):
    """ Dumps an object into a pickle file.

    Args:
        obj: object to pickle
        filepath: destination path for the pickle file.
    """
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)


def load_config(module_path, config_name):
    """Load a config class from a python module.

    Args:
        module_path: path to the module containing the config class.
        config_name: name of the config class.

    Returns:
        A config class.
    """

    module = import_module(module_path)
    return getattr(module, config_name)


class DatasetConfigLoader:
    FIQA = 'FiQAConfig'
    MSM = 'MSMConfig'
    WikiPQA = 'WikiConfig'
    INSURANCE_QA = 'InsuranceConfig'

    def get_dataset_config(self, config_module, dataset_name):
        if dataset_name == 'FiQA':
            return load_config(config_module, self.FIQA)
        elif dataset_name == 'MSmarco':
            return load_config(config_module, self.MSM)
        elif dataset_name == 'WikipassageQA':
            return load_config(config_module, self.WikiPQA)
        elif dataset_name == 'InsuranceQA':
            return load_config(config_module, self.INSURANCE_QA)
        raise NotImplementedError()
