import h5py
import numpy as np

from qa_utils.io import dump_pkl_file
from qa_utils.preprocessing.dataset import Dataset, Trainset, Testset
from qa_utils.preprocessing.hdf5saver import Hdf5Saver
from qa_utils.text import compute_idfs


class DuetHhdf5Saver(Hdf5Saver):
    """Class for transforming and saving a qa_utils dataset into an hdf5 file that matches the input specification of
    DUET V2.
    """

    def __init__(self, dataset: Dataset, tokenizer, max_vocab_size, vocab_outfile, idf_outfile, max_query_len,
                 max_doc_len, train_outfile=None, dev_outfile=None, test_outfile=None):
        """Construct a hdf5 saver for qa_util Datasets.

        Args:
            dataset: the dataset to save as hdf5.
            max_query_len: truncate queries to this length in words.
            max_doc_len: truncate documents to this length in words.
            vocab_outfile: a pickle file where a word to index dictionary will be exported to.
            max_vocab_size: the maximum number of words in the vocabulary, only keeping the most frequent ones. Uses all
            if None.
        """

        super().__init__(dataset, tokenizer, max_vocab_size, vocab_outfile, train_outfile, dev_outfile, test_outfile)

        # compute idfs for weighting of the interaction matrix
        vocab_tokens = set(self.word_to_index.keys())
        self.idfs = compute_idfs(vocab_tokens, dataset.docs.values(), self.tokenizer)
        # map token ids to idfs
        self.idfs = dict(map(lambda x: (self.word_to_index[x[0]], x[1]), self.idfs.items()))
        dump_pkl_file(self.idfs, idf_outfile)

        print('tokenizing...')
        self.dataset.transform_docs(lambda x: self._words_to_index(self.tokenizer.tokenize(x))[:max_doc_len])
        self.dataset.transform_queries(lambda x: self._words_to_index(self.tokenizer.tokenize(x)[:max_query_len]))

    def _define_trainset(self, dataset_fp, n_out_examples):
        vlen_int64 = h5py.special_dtype(vlen=np.dtype('int64'))
        dataset_fp.create_dataset('queries', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('pos_docs', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('neg_docs', shape=(n_out_examples,), dtype=vlen_int64)

    def _define_candidate_set(self, dataset_fp, n_out_examples):
        vlen_int64 = h5py.special_dtype(vlen=np.dtype('int64'))
        dataset_fp.create_dataset('queries', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('docs', shape=(n_out_examples,), dtype=vlen_int64)

        dataset_fp.create_dataset('q_ids', shape=(n_out_examples,), dtype=np.dtype('int64'))
        dataset_fp.create_dataset('labels', shape=(n_out_examples,), dtype=np.dtype('int64'))

    def _n_out_samples(self, dataset):
        if isinstance(dataset, Trainset):
            # duet trains on triplets i.e. pairs of (q, pos, neg)
            return len(dataset.pos_pairs) * dataset.num_neg_examples
        elif isinstance(dataset, Testset):
            return len(dataset)
        else:
            raise TypeError('Dataset needs to be of type Trainset or Testset.')

    def _save_train_row(self, fp, query, pos_doc, neg_docs, idx):
        for neg_ids in neg_docs:
            fp['queries'][idx] = query
            fp['pos_docs'][idx] = pos_doc
            fp['neg_docs'][idx] = neg_ids

    def _save_candidate_row(self, fp, q_id, query, doc, label, idx):
        fp['q_ids'][idx] = q_id
        fp['queries'][idx] = query
        fp['docs'][idx] = doc
        fp['labels'][idx] = label

    def _words_to_index(self, words, unknown_token='<UNK>'):
        """Turns a list of words into integer indices using self.word_to_index.

        Args:
            words (list(str)): list of words.

        Returns:
            list(int): a list if integers encoding words.
        """
        tokens = []
        for token in words:
            try:
                tokens.append(self.word_to_index[token])
            # out of vocabulary
            except KeyError:
                tokens.append(self.word_to_index[unknown_token])
        return tokens
