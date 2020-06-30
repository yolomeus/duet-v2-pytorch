import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchtext import vocab

from data_source import DuetHdf5Trainset, DuetHdf5Testset
from qa_utils.io import load_pkl_file, load_json_file
from qa_utils.lightning import BaseRanker


class DuetV2(BaseRanker):
    """Implementation of the DuetV2 model.
    """

    def __init__(self, lr, batch_size, id_to_word_file, idfs_file, glove_name, glove_cache, glove_dim, h_dim, max_q_len,
                 max_d_len, dropout_rate, train_file, val_file, test_file, pooling_size_doc=100):
        """

        Args:
            id_to_word_file: a mapping from all integer ids in the vocabulary to tokens.
            glove_name: version of the pre-trained glove vectors to use. One of: ['42B', '840B', 'twitter.27B', '6B']
            glove_cache: the directory to download the glove vectors to.
            h_dim: Hidden dimension across the network.
            max_q_len: input length of queries.
            max_d_len: input length of documents.
            dropout_rate: dropout rate for all dropout layers.
            pooling_size_doc: size of the max pooling window for the document after convolution.
        """
        id_to_word = load_json_file(id_to_word_file)
        idfs = load_pkl_file(idfs_file)

        train_ds = DuetHdf5Trainset(train_file, max_q_len, max_d_len, idfs)
        val_ds = DuetHdf5Testset(val_file, max_q_len, max_d_len, idfs)
        test_ds = DuetHdf5Testset(test_file, max_q_len, max_d_len, idfs)
        super().__init__(train_ds, val_ds, test_ds, batch_size)

        self.local_model = DuetV2Local(h_dim, max_q_len, max_d_len, dropout_rate)
        self.distributed_model = DuetV2Distributed(id_to_word,
                                                   glove_name,
                                                   glove_cache,
                                                   glove_dim,
                                                   h_dim,
                                                   dropout_rate,
                                                   pooling_size_doc,
                                                   max_q_len - 2,
                                                   max_d_len)

        # combining layers
        self.linear_0 = nn.Linear(h_dim, h_dim)
        self.linear_1 = nn.Linear(h_dim, h_dim)
        self.linear_out = nn.Linear(h_dim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.ce_loss = CrossEntropyLoss()
        self.lr = lr
        self.batch_size = batch_size

        # don't save the datasets
        self.save_hyperparameters()

    def forward(self, inputs):
        """Run the forward pass.

        Args:
            inputs:
                - query: a fixed number of integer id's.
                - doc: a fixed number of integer id's.
                - imat: a (max_q_len x max_d_len) interaction matrix.
        """

        query, doc, imat = inputs
        local = self.local_model(imat)
        dist = self.distributed_model(query, doc)

        x = dist + local
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear_out(x)

        return x * 0.1

    def training_step(self, batch, batch_idx):
        pos_batch, neg_batch, y_batch = batch
        pos_out, neg_out = self(pos_batch), self(neg_batch)
        out = torch.cat([pos_out, neg_out], 1)
        loss = self.ce_loss(out, y_batch)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(ap):
        ap.add_argument('IDF_FILE', help='JSON file containing the mapping from ids to words.')
        ap.add_argument('--glove_name', default='840B', help='GloVe embedding name')
        ap.add_argument('--glove_cache', default='glove_cache', help='Glove cache directory.')

        ap.add_argument('--glove_dim', type=int, default=300, help='The dimensionality of the GloVe embeddings')
        ap.add_argument('--hidden_dim', type=int, default=300,
                        help='The hidden dimension used throughout the whole network.')

        ap.add_argument('--max_q_len', type=int, default=20, help='Maximum query length.')
        ap.add_argument('--max_d_len', type=int, default=200, help='Maximum document legth.')

        ap.add_argument('--batch_size', type=int, default=32, help='Batch size')
        ap.add_argument('--dropout', type=float, default=0.5, help='Dropout value')
        ap.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
        return ap


class DuetV2Local(torch.nn.Module):
    """The local part of the Duet model which is trained on a query - document interaction matrix.
    """

    def __init__(self, h_dim, max_q_len, max_d_len, dropout_rate):
        """Constructs the local duet module.

        Args:
            h_dim (int): hidden dimension across the network.
            max_q_len (int): the maximum number of tokens in the query.
            max_d_len (int): the maximum number of tokens in the document.
            dropout_rate (float): dropout rate for all dropout layers.
        """
        super().__init__()

        self.conv1d = nn.Conv1d(max_d_len, h_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.flatten = nn.Flatten()
        self.linear_0 = nn.Linear(h_dim * max_q_len, h_dim)
        self.linear_1 = nn.Linear(h_dim, h_dim)

    def forward(self, imat):
        x = self.conv1d(imat)
        x = self.relu(x)
        x = self.flatten(x)

        x = self.dropout(x)
        x = self.linear_0(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.relu(x)

        return self.dropout(x)


class DuetV2Distributed(torch.nn.Module):
    """The distributed part of the duet model. It is Trained on distributed representations of query and document i.e.
    word embeddings.
    """

    def __init__(self, id_to_word, glove_name, glove_cache, glove_dim, h_dim, dropout, pooling_size_doc,
                 pooling_size_query,
                 max_d_len):
        """

        Args:
            id_to_word: a mapping from all integer ids in the vocabulary to tokens.
            glove_name: version of the pre-trained glove vectors to use. One of: ['42B', '840B', 'twitter.27B', '6B']
            glove_cache: the directory to download the glove vectors to.
            h_dim: Hidden dimension across the network.
            dropout: dropout rate for all dropout layers.
            pooling_size_doc: size of the max pooling window for the document after convolution.
            pooling_size_query: size of the max pooling window for the query after convolution.
            max_d_len: input length of documents.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.glove = GloveEmbedding(id_to_word, glove_name, glove_dim, cache=glove_cache, padding_idx=0)
        self.conv1d_query = nn.Conv1d(glove_dim, h_dim, kernel_size=3)
        self.linear_query = nn.Linear(h_dim, h_dim)
        self.max_pool_query = nn.MaxPool1d(pooling_size_query)

        self.conv1d_doc_0 = nn.Conv1d(glove_dim, h_dim, kernel_size=3)
        self.conv1d_doc_1 = nn.Conv1d(h_dim, h_dim, kernel_size=1)
        self.max_pool_doc = nn.MaxPool1d(pooling_size_doc, stride=1)

        n_pooling_windows = max_d_len - pooling_size_doc - 1
        self.comb_linear_0 = nn.Linear(h_dim * n_pooling_windows, h_dim)
        self.comb_linear_1 = nn.Linear(h_dim, h_dim)

    def forward(self, query, doc):
        query_embeds = self.glove(query).permute(0, 2, 1)  # swap channel with time dimension for conv1d
        doc_embeds = self.glove(doc).permute(0, 2, 1)

        q = self.conv1d_query(query_embeds)
        q = self.relu(q)
        q = self.max_pool_query(q)
        q = self.flatten(q)
        q = self.linear_query(q)
        q = self.relu(q)

        d = self.conv1d_doc_0(doc_embeds)
        d = self.relu(d)
        d = self.max_pool_doc(d)
        d = self.conv1d_doc_1(d)
        d = self.relu(d)

        x = q.unsqueeze(-1) * d
        x = self.flatten(x)
        x = self.dropout(x)

        x = self.comb_linear_0(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.comb_linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class GloveEmbedding(torch.nn.Module):
    def __init__(self, id_to_word, name, dim, freeze=False, cache=None, padding_idx=None):
        super().__init__()
        self.id_to_word = id_to_word
        self.name = name
        self.dim = dim
        self.padding_idx = padding_idx
        self.glove = vocab.GloVe(name=name, dim=dim, cache=cache)
        weights = self._get_weights()
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze)

    def _get_weights(self):
        weights = []
        i = 0
        for idx in sorted(self.id_to_word):
            word = self.id_to_word[idx]
            if i == self.padding_idx:
                weights.append(torch.zeros([self.dim]))
                i += 1
                continue
            if word in self.glove.stoi:
                glove_idx = self.glove.stoi[word]
                weights.append(self.glove.vectors[glove_idx])
                i += 1
            else:
                # initialize randomly
                weights.append(torch.zeros([self.dim]).uniform_(-0.25, 0.25))
        print(f'Imported {i} words from GloVe')
        # this converts a list of tensors to a new tensor
        return torch.stack(weights)

    def forward(self, x):
        return self.embedding(x)
