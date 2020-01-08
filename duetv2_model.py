import torch
from torch import nn


class DuetV2(torch.nn.Module):
    def __init__(self, num_embeddings, h_dim, max_q_len, max_d_len, dropout_rate, out_features):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, h_dim)

        self.local_model = DuetV2Local(h_dim, max_q_len, max_d_len, dropout_rate)

        self.linear_out = nn.Linear(h_dim, out_features)

    def forward(self, query, doc, imat):
        # query_embed = self.embedding(query)
        # doc_embed = self.embedding(doc)
        x = self.local_model(imat)
        return self.linear_out(x)


class DuetV2Local(torch.nn.Module):
    """The local part of the Duet model which is trained on a query - document interaction matrix.
    """

    def __init__(self, h_dim, max_q_len, max_d_len, dropout_rate):
        super().__init__()

        self.conv1d = nn.Conv1d(max_d_len, h_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.flatten = nn.Flatten()
        self.linear_0 = nn.Linear(h_dim * max_q_len, h_dim)
        self.linear_1 = nn.Linear(h_dim, h_dim)

    def forward(self, imat):
        x = torch.transpose(imat, -1, -2)
        x = self.conv1d(x)
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
    pass
