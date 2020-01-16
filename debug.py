from pprint import pprint

from data_source import DuetHdf5Testset, DuetHdf5Trainset
from qa_utils.io import load_pkl_file
from qa_utils.text import indices_to_words


def print_testset(n, vocab='data/fiqa/vocabulary.pkl', hdf_path='data/fiqa/dev.hdf5', max_q=20, max_d=200):
    data = DuetHdf5Testset(hdf_path, max_q, max_d)
    itow = load_pkl_file(vocab)
    i = 0
    for qid, (qids, docids, imat), label in data:
        print("query:")
        pprint(' '.join(indices_to_words(qids, itow)))
        print("doc:")
        pprint(' '.join(indices_to_words(docids, itow)))
        print('label:', label)
        print('--------------------------------------------------------------------')

        if i == n:
            break
        i += 1


def print_trainset(n, vocab='data/fiqa/vocabulary.pkl', hdf_path='data/fiqa/train.hdf5', max_q=20, max_d=200):
    data = DuetHdf5Trainset(hdf_path, max_q, max_d)
    itow = load_pkl_file(vocab)
    i = 0
    for (qids, docids, pos_imat), (queries, pos_doc, pos_imat), label in data:
        print("query:")
        pprint(' '.join(indices_to_words(qids, itow)))
        print("doc:")
        pprint(' '.join(indices_to_words(docids, itow)))
        print('label:', label)
        print('--------------------------------------------------------------------')

        if i == n:
            break
        i += 1


if __name__ == '__main__':
    print_trainset(10)
