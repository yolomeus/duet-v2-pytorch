import pickle
from abc import ABC, abstractmethod

import h5py
import numpy as np
from tqdm import tqdm

from config import FiQADatasetConfig
from preprocessing import tokenizers
from qa_utils.preprocessing.dataset import Dataset
from qa_utils.preprocessing.fiqa import FiQA
from util.io import dump_pkl_file
from util.text import build_vocab, compute_idfs


class Hdf5Saver(ABC):
    """Saves a dataset to hdf5 format.
    """

    def __init__(self, dataset: Dataset, train_outfile=None, dev_outfile=None, test_outfile=None):
        """Construct a h5py saver object. Each dataset that has no output path specified will be ignored, meaning at
        least one output path must be provided.

        Args:
            dataset: Dataset to save to hdf5.
            train_outfile: path to the hdf5 output file for the train set.
            dev_outfile: path to the hdf5 output file for the dev set.
            test_outfile: path to the hdf5 output file for the test set.
        """
        self.dataset = dataset

        self.train_outfile = train_outfile
        self.dev_outfile = dev_outfile
        self.test_outfile = test_outfile

        out_paths = [train_outfile, dev_outfile, test_outfile]
        assert any(out_paths), 'you need to specify at least one output filepath.'
        self.train_out, self.dev_out, self.test_out = (h5py.File(fpath, 'w') if fpath else None for fpath in
                                                       out_paths)

    @abstractmethod
    def _save_row(self, query, pos_doc, neg_docs, h5py_fp):
        """Apply any necessary transformation to the input data and save it to the h5py file.

        Args:
            query: a raw text query to be saved.
            pos_doc: a raw text document relevant to the query.
            neg_docs: One or more documents that are not relevant to the query.
            h5py_fp: file pointer to the hdf5 output file

        """

    @abstractmethod
    def _define_dataset(self, dataset_fp, n_out_examples):
        """Specify the structure of the hdf5 output file.

        Args:
            dataset_fp: file pointer to the hdf5 output file.
            n_out_examples: number of examples that will be generated for this dataset.
        """

    @abstractmethod
    def output_size(self, split):
        """Computes the number of output examples generated for either train or test set.

        Args:
            split: either 'train' or 'test'

        Returns:
            int: number of total output

        """
        assert split in ['train', 'test'], 'can only compute output_size for either "train" or "test".'

    def _save_trainset(self):
        print("saving", self.train_outfile, "...")
        self._define_dataset(self.train_out, self.output_size('train'))
        self.idx = 0
        for query, pos_doc, neg_docs in tqdm(self.dataset.trainset):
            self._save_row(query, pos_doc, neg_docs, self.train_out)


class DuetHhdf5Saver(Hdf5Saver):

    def __init__(self, dataset: Dataset, max_query_len, max_doc_len, vocab_outfile, *args, max_vocab_size=None,
                 **kwargs):
        """Construct a hdf5 saver for qa_util Datasets.

        Args:
            dataset: the dataset to save as hdf5.
            max_query_len: truncate queries to this length in words.
            max_doc_len: truncate documents to this length in words.
            vocab_outfile: a pickle file where a word to index dictionary will be exported to.
            max_vocab_size: the maximum number of words in the vocabulary, only keeping the most frequent ones. Uses all
            if None.
        """
        super().__init__(dataset, *args, **kwargs)
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

        self.tokenizer = tokenizers.DuetTokenizer()

        collection = list(dataset.queries.values()) + list(dataset.docs.values())
        self.word_to_index = build_vocab(collection, self.tokenizer, max_vocab_size=max_vocab_size)
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        dump_pkl_file(self.index_to_word, vocab_outfile)

        # compute idfs for weighting of the interaction matrix
        vocab_tokens = set(self.word_to_index.keys())
        self.idfs = compute_idfs(vocab_tokens, dataset.docs.values(), self.tokenizer)
        # map token ids to idfs
        self.idfs = dict(map(lambda x: (self.word_to_index[x[0]], x[1]), self.idfs.items()))

    def output_size(self, split):
        if split == 'train':
            trainset = self.dataset.trainset
            # duet trains on triplets i.e. one pairs of (q, pos, neg)
            return len(trainset.pos_pairs) * trainset.num_neg_examples
        elif split == 'test':
            pass

    def _save_row(self, query, pos_doc, neg_docs, h5py_fp):
        q_tokens = self.tokenizer.tokenize(query)
        q_ids = self._words_to_index(q_tokens)[:self.max_query_len]

        pos_tokens = self.tokenizer.tokenize(pos_doc)
        pos_ids = self._words_to_index(pos_tokens)[:self.max_doc_len]

        neg_docs_ids = [self._words_to_index(self.tokenizer.tokenize(neg_doc)) for neg_doc in neg_docs]

        pos_imat = self._build_interaction_matrix(q_ids, pos_ids, self.idfs)

        for neg_ids in neg_docs_ids:
            neg_ids = neg_ids[:self.max_doc_len]

            h5py_fp['queries'][self.idx] = q_ids
            h5py_fp['pos_docs'][self.idx] = pos_ids
            h5py_fp['neg_docs'][self.idx] = neg_ids

            neg_imat = self._build_interaction_matrix(q_ids, neg_ids, self.idfs)

            h5py_fp['pos_imats'][self.idx] = pos_imat
            h5py_fp['neg_imats'][self.idx] = neg_imat
            self.idx += 1

    def _build_interaction_matrix(self, q_ids, doc_ids, idfs):
        """Helper method for building a IDF-weighted interaction matrix between a query and document. An entry at (i, j)
        is the IDF of the i-th word in the query if it matches the j-th word in the document or 0 else.

        Args:
            q_ids (Iterable): integer ids representing the words in the query.
            doc_ids (Iterable): integer ids representing the words in the document.
            idfs (dict): mapping from each id in the vocabulary of query and document to an IDF value.
        Returns:
            numpy.ndarray: A 2-D interaction matrix.
        """
        m = np.zeros(shape=(self.max_query_len, self.max_doc_len))
        for i in range(len(q_ids)):
            for j in range(len(doc_ids)):
                cur_qid = q_ids[i]
                if cur_qid == doc_ids[j]:
                    m[i, j] = idfs[cur_qid]

        return m

    def _define_dataset(self, dataset_fp, n_out_examples):
        vlen_uint32 = h5py.special_dtype(vlen=np.dtype('uint32'))
        dataset_fp.create_dataset('queries', shape=(n_out_examples,), dtype=vlen_uint32)
        dataset_fp.create_dataset('pos_docs', shape=(n_out_examples,), dtype=vlen_uint32)
        dataset_fp.create_dataset('neg_docs', shape=(n_out_examples,), dtype=vlen_uint32)

        imat_shape = (n_out_examples, self.max_query_len, self.max_doc_len)
        dataset_fp.create_dataset('pos_imats', shape=imat_shape, dtype=np.dtype('float32'))
        dataset_fp.create_dataset('neg_imats', shape=imat_shape, dtype=np.dtype('float32'))

    def _words_to_index(self, words):
        """Turns a list of words into integer indices using self.word_to_index.

        Args:
            words (list(str)): list of words.

        Returns:
            list(int): a list if integers encoding words.
        """
        return [self.word_to_index[token] for token in words]


if __name__ == '__main__':
    conf = FiQADatasetConfig()
    fiqa = FiQA(args=conf)
    saver = DuetHhdf5Saver(fiqa,
                           20,
                           200,
                           './data/fiqa/vocabulary.pkl',
                           './data/fiqa/train.hdf5',
                           './data/fiqa/dev.hdf5',
                           './data/fiqa/test.hdf5')
    saver._save_trainset()
