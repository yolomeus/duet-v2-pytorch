from pprint import pprint

from data_source import DuetHdf5Trainset
from duet_utils.io import load_pkl_file, DatasetConfigLoader
from duet_utils.text import indices_to_words


def print_decoded_dataset(dataset_name, vocab_file, n_samples, config_path='config'):
    dataset_config = DatasetConfigLoader().get_dataset_config(config_path, dataset_name)
    trainset = DuetHdf5Trainset(dataset_config.TRAIN_HDF5_PATH)
    index_to_word = load_pkl_file(vocab_file)
    i = 0
    for queries, pos_doc, neg_docs, pos_imats, neg_imats in trainset:
        print('query:')
        pprint(' '.join(indices_to_words(queries, index_to_word)))
        print('-------------------------------------------')
        print('pos_doc:')
        pprint(' '.join(indices_to_words(pos_doc, index_to_word)))
        print('-------------------------------------------')
        print('neg_doc:')
        pprint(' '.join(indices_to_words(neg_docs, index_to_word)))
        print('-------------------------------------------')
        i += 1
        if i == n_samples:
            break
