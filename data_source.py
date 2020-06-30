import h5py
import numpy as np
from torch.utils import data


class DuetHdf5Dataset(data.Dataset):
    def __init__(self, file_path, max_query_len, max_doc_len, idfs):
        self.fp = file_path
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.idfs = idfs

        with h5py.File(self.fp, 'r') as fp:
            self.length = len(fp['queries'])

    @staticmethod
    def _pad_to(x, n):
        return np.pad(x, (0, n - len(x)), constant_values=0)

    def _build_interaction_matrix(self, q_ids, doc_ids):
        m = np.zeros(shape=(self.max_doc_len, self.max_query_len), dtype=np.float32)
        for j in range(len(doc_ids)):
            for i in range(len(q_ids)):

                cur_qid = q_ids[i]
                if cur_qid == doc_ids[j]:
                    m[j, i] = self.idfs[cur_qid]

        return m


class DuetHdf5Trainset(DuetHdf5Dataset):

    def __getitem__(self, index):
        with h5py.File(self.fp, 'r') as fp:
            query = fp['queries'][index][:self.max_query_len]
            pos_doc = fp['pos_docs'][index][:self.max_doc_len]
            neg_doc = fp['neg_docs'][index][:self.max_doc_len]

            queries = self._pad_to(query, self.max_query_len)

            pos_docs = self._pad_to(pos_doc, self.max_doc_len)
            neg_docs = self._pad_to(neg_doc, self.max_doc_len)

            pos_imat = self._build_interaction_matrix(query, pos_doc)
            pos_sample = queries, pos_docs, pos_imat

            neg_imat = self._build_interaction_matrix(query, neg_doc)
            neg_sample = queries, neg_docs, neg_imat

        return pos_sample, neg_sample, 0

    def __len__(self):
        return self.length


class DuetHdf5Testset(DuetHdf5Dataset):

    def __getitem__(self, index):
        with h5py.File(self.fp, 'r') as fp:
            qids = fp['q_ids'][index]
            inputs = (self._pad_to(fp['queries'][index], self.max_query_len),
                      self._pad_to(fp['docs'][index], self.max_doc_len),
                      self._build_interaction_matrix(fp['queries'][index], fp['docs'][index]))
            labels = fp['labels'][index]

            return qids, inputs, labels

    def __len__(self):
        return self.length
