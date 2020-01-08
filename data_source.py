import h5py
import numpy as np
from torch.utils import data


class DuetHdf5Dataset(data.Dataset):
    def __init__(self, file_path, max_query_len, max_doc_len):
        self.fp = h5py.File(file_path, 'r')
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    @staticmethod
    def _pad_to(x, n):
        return np.pad(x, (0, n - len(x)), constant_values=0)


class DuetHdf5Trainset(DuetHdf5Dataset):
    def __init__(self, file_path, max_query_len, max_doc_len):
        super().__init__(file_path, max_query_len, max_doc_len)

        fp = self.fp

        self.queries = fp['queries']

        self.pos_docs = fp['pos_docs']
        self.neg_docs = fp['neg_docs']

        self.pos_imats = fp['pos_imats']
        self.neg_imats = fp['neg_imats']

    def __getitem__(self, index):
        queries = self._pad_to(self.queries[index], self.max_query_len)

        pos_docs = self._pad_to(self.pos_docs[index], self.max_doc_len)
        neg_docs = self._pad_to(self.neg_docs[index], self.max_doc_len)

        pos_sample = queries, pos_docs, self.pos_imats[index]
        neg_sample = queries, neg_docs, self.neg_imats[index]

        return pos_sample, neg_sample, 0

    def __len__(self):
        return len(self.queries)
