import torch
from torch import nn


class DuetV2(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        super().__init__()
        h_dim = 500
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.linear = nn.Linear(32160, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)
        self.linear3 = nn.Linear(h_dim, h_dim)
        self.linear4 = nn.Linear(h_dim, h_dim)
        self.out = nn.Linear(h_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, query, doc, imat):
        query_embed = self.embedding(query)
        doc_embed = self.embedding(doc)
        flattened_inputs = [x.view(x.size(0), -1) for x in [query_embed, doc_embed, imat]]
        x = torch.cat(flattened_inputs, -1)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class DuetV2Local(torch.nn.Module):
    """The local part of the Duet model which is trained on a query - document interaction matrix.
    """
    pass


class DuetV2Distributed(torch.nn.Module):
    """The distributed part of the duet model. It is Trained on distributed representations of query and document i.e.
    word embeddings.
    """
    pass
