from abc import ABC, abstractmethod

import h5py
import numpy as np
from tqdm import tqdm

from config import FiQAConfig
from preprocessing.tokenizer import DuetTokenizer

from qa_utils.preprocessing.dataset import Dataset, Trainset, Testset
from qa_utils.preprocessing.fiqa import FiQA
from qa_utils.io import dump_pkl_file
from qa_utils.text import build_vocab, compute_idfs


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

        self.train_outpath = train_outfile
        self.dev_outpath = dev_outfile
        self.test_outpath = test_outfile

        out_paths = [train_outfile, dev_outfile, test_outfile]
        assert any(out_paths), 'you need to specify at least one output filepath.'
        self.train_out, self.dev_out, self.test_out = (h5py.File(fpath, 'w') if fpath else None for fpath in
                                                       out_paths)

    def build_all(self):
        """Exports each split of dataset to hdf5 if an output file was specified for it.
        """
        if self.train_out:
            self._save_train_set()
        if self.test_out:
            self._save_candidate_set('test')
        if self.dev_out:
            self._save_candidate_set('dev')

    def _save_candidate_set(self, split):
        """Saves a candidate type set i.e. Dataset.testset or Dataset.devset to hdf5.

        Args:
            split (str): either 'dev' or 'test'.

        """
        fp, dataset = (self.dev_out, self.dataset.devset) if split == 'dev' else (self.test_out, self.dataset.testset)

        print('saving to', fp.filename, '...')
        self._define_candidate_set(fp, self._n_out_samples(dataset))
        self.idx = 0

        for q_id, query, doc, label in tqdm(dataset):
            processed_row = self._transform_candidate_row(q_id, query, doc, label)
            self._save_candidate_row(fp, *processed_row)

    def _save_train_set(self):
        """Saves the trainset to hdf5.
        """
        print("saving", self.train_outpath, "...")
        self._define_trainset(self.train_out, self._n_out_samples(self.dataset.trainset))
        self.idx = 0

        for query, pos_doc, neg_docs in tqdm(self.dataset.trainset):
            processed_row = self._transform_train_row(query, pos_doc, neg_docs)
            self._save_train_row(*processed_row)

    @abstractmethod
    def _define_trainset(self, dataset_fp, n_out_examples):
        """Specify the structure of the hdf5 output file.

        Args:
            dataset_fp: file pointer to the hdf5 output file.
            n_out_examples: number of examples that will be generated for this dataset.
        """

    @abstractmethod
    def _define_candidate_set(self, dataset_fp, n_out_examples):
        """Specify the structure of the hdf5 output file for candidate type sets.

        Args:
            dataset_fp: file pointer to the hdf5 output file.
            n_out_examples: number of examples that will be generated for this dataset.
        """

    @abstractmethod
    def _n_out_samples(self, dataset):
        """Computes the number of output examples generated for either train, test or dev set.

        Args:
            dataset: either Trainset or Testset from qa_utils.

        Returns:
            int: number of total output samples.

        """

    @abstractmethod
    def _transform_train_row(self, query, pos_doc, neg_docs):
        """This function is applied to each row in the train set as returned by a qa_utils Trainset before saving.

        Args:
            query (str): a query string.
            pos_doc (str): a positive document string w.r.t. the query.
            neg_docs (list(str)): a list of negative documents.

        Returns:
            The transformed row.
        """

    @abstractmethod
    def _transform_candidate_row(self, q_id, query, doc, label):
        """This function is applied to each row in the test ord dev set as returned by qa_utils Trainset before saving.

                Args:

                Returns:
                    The transformed row.
        """

    @abstractmethod
    def _save_train_row(self, *args):
        """The function that saves an item from the Dataset.trainset after applying _transform_train_row. It's saved to
        a hdf5 file as defined in _define_trainset().

        Args:
            *args: the transformed row returned by _transform_train_row.
        """

    @abstractmethod
    def _save_candidate_row(self, *args):
        """The function that saves an item from the Dataset.devset or Dataset.trainset after applying
        _transform_candidate_row. It's saved a hdf5 file as defined in _define_candidate_set().

        Args:
            *args: he transformed row returned by _transform_candidate_row.
        """


class DuetHhdf5Saver(Hdf5Saver):
    """Class for transforming and saving a qa_utils dataset into an hdf5 file that matches the input specification of
    DUET V2.
    """

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

        self.tokenizer = DuetTokenizer()

        collection = list(dataset.queries.values()) + list(dataset.docs.values())
        self.word_to_index = build_vocab(collection, self.tokenizer, max_vocab_size=max_vocab_size)
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        dump_pkl_file(self.index_to_word, vocab_outfile)

        # compute idfs for weighting of the interaction matrix
        vocab_tokens = set(self.word_to_index.keys())
        self.idfs = compute_idfs(vocab_tokens, dataset.docs.values(), self.tokenizer)
        # map token ids to idfs
        self.idfs = dict(map(lambda x: (self.word_to_index[x[0]], x[1]), self.idfs.items()))

    def _define_trainset(self, dataset_fp, n_out_examples):
        vlen_int64 = h5py.special_dtype(vlen=np.dtype('int64'))
        dataset_fp.create_dataset('queries', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('pos_docs', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('neg_docs', shape=(n_out_examples,), dtype=vlen_int64)

        imat_shape = (n_out_examples, self.max_query_len, self.max_doc_len)
        dataset_fp.create_dataset('pos_imats', shape=imat_shape, dtype=np.dtype('float32'))
        dataset_fp.create_dataset('neg_imats', shape=imat_shape, dtype=np.dtype('float32'))

    def _define_candidate_set(self, dataset_fp, n_out_examples):
        vlen_int64 = h5py.special_dtype(vlen=np.dtype('int64'))
        dataset_fp.create_dataset('queries', shape=(n_out_examples,), dtype=vlen_int64)
        dataset_fp.create_dataset('docs', shape=(n_out_examples,), dtype=vlen_int64)

        imat_shape = (n_out_examples, self.max_query_len, self.max_doc_len)
        dataset_fp.create_dataset('imats', shape=imat_shape, dtype=np.dtype('float32'))

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

    def _transform_train_row(self, query, pos_doc, neg_docs):
        q_tokens = self.tokenizer.tokenize(query)
        q_ids = self._words_to_index(q_tokens)[:self.max_query_len]

        pos_tokens = self.tokenizer.tokenize(pos_doc)
        pos_ids = self._words_to_index(pos_tokens)[:self.max_doc_len]

        neg_docs_ids = [self._words_to_index(self.tokenizer.tokenize(neg_doc))[:self.max_doc_len] for neg_doc in
                        neg_docs]

        pos_imat = self._build_interaction_matrix(q_ids, pos_ids)
        neg_imats = []

        for neg_ids in neg_docs_ids:
            neg_imat = self._build_interaction_matrix(q_ids, neg_ids)
            neg_imats.append(neg_imat)

        return q_ids, pos_ids, neg_docs_ids, pos_imat, neg_imats

    def _save_train_row(self, q_ids, pos_ids, neg_docs_ids, pos_imat, neg_imats):
        fp = self.train_out
        for neg_ids, neg_imat in zip(neg_docs_ids, neg_imats):
            fp['queries'][self.idx] = q_ids
            fp['pos_docs'][self.idx] = pos_ids
            fp['neg_docs'][self.idx] = neg_ids

            fp['pos_imats'][self.idx] = pos_imat
            fp['neg_imats'][self.idx] = neg_imat
            self.idx += 1

    def _transform_candidate_row(self, q_id, query, doc, label):
        q_tokens = self.tokenizer.tokenize(query)
        q_ids = self._words_to_index(q_tokens)[:self.max_query_len]

        doc_tokens = self.tokenizer.tokenize(doc)
        doc_ids = self._words_to_index(doc_tokens)[:self.max_doc_len]

        imat = self._build_interaction_matrix(q_ids, doc_ids)

        return q_id, q_ids, doc_ids, imat, label

    def _save_candidate_row(self, fp, q_id, q_ids, doc_ids, imat, label):
        fp['q_ids'][self.idx] = q_id

        fp['queries'][self.idx] = q_ids
        fp['docs'][self.idx] = doc_ids
        fp['imats'][self.idx] = imat

        fp['labels'][self.idx] = label

        self.idx += 1

    def _build_interaction_matrix(self, q_ids, doc_ids):
        """Helper method for building a IDF-weighted interaction matrix between a query and document. An entry at (i, j)
        is the IDF of the i-th word in the query if it matches the j-th word in the document or 0 else.

        Args:
            q_ids (Iterable): integer ids representing the words in the query.
            doc_ids (Iterable): integer ids representing the words in the document.
        Returns:
            numpy.ndarray: A 2-D interaction matrix.
        """
        m = np.zeros(shape=(self.max_query_len, self.max_doc_len))
        for i in range(len(q_ids)):
            for j in range(len(doc_ids)):
                cur_qid = q_ids[i]
                if cur_qid == doc_ids[j]:
                    m[i, j] = self.idfs[cur_qid]

        return m

    def _words_to_index(self, words):
        """Turns a list of words into integer indices using self.word_to_index.

        Args:
            words (list(str)): list of words.

        Returns:
            list(int): a list if integers encoding words.
        """
        return [self.word_to_index[token] for token in words]


if __name__ == '__main__':
    # TODO proper script for data generation
    conf = FiQAConfig()
    fiqa = FiQA(args=conf)
    saver = DuetHhdf5Saver(fiqa,
                           20,
                           200,
                           './data/fiqa/vocabulary.pkl',
                           './data/fiqa/train.hdf5',
                           './data/fiqa/dev.hdf5',
                           './data/fiqa/test.hdf5')
    saver.build_all()
