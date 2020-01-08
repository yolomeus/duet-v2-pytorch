from argparse import ArgumentParser

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_source import DuetHdf5Trainset
from duet_utils.training import train_model_pairwise
from duetv2_model import DuetV2

if __name__ == '__main__':
    ap = ArgumentParser(description='Train the DUET model.')
    ap.add_argument('-TRAIN_DATA', help='Path to an hdf5 file containing the training data.')
    ap.add_argument('-VOCAB_SIZE', type=int, help='Size of the vocabulary in the training file.')

    ap.add_argument('--max_q_len', type=int, default=20, help='Maximum query length.')
    ap.add_argument('--max_d_len', type=int, default=200, help='Maximum document legth.')

    ap.add_argument('--hidden_dim', type=int, default=300,
                    help='The hidden dimension used throughout the whole network.')
    ap.add_argument('--dropout', type=float, default=0.5, help='Dropout value')
    ap.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')

    ap.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    ap.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    ap.add_argument('--accumulate_batches', type=int, default=1,
                    help='Update weights after this many batches')
    ap.add_argument('--working_dir', default='train', help='Working directory for checkpoints and logs')

    args = ap.parse_args()

    trainset = DuetHdf5Trainset(args.TRAIN_DATA, args.max_q_len, args.max_d_len)
    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    if torch.cuda.is_available():
        # cuda:0 will still use all GPUs
        device = torch.device('cuda:0')
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print('using {} device(s): "{}"'.format(torch.cuda.device_count(), dev_name))
    else:
        device = torch.device('cpu')

    model = DuetV2(num_embeddings=args.VOCAB_SIZE,
                   h_dim=args.hidden_dim,
                   max_q_len=args.max_q_len,
                   max_d_len=args.max_d_len,
                   dropout_rate=args.dropout,
                   out_features=1)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_model_pairwise(model, train_dataloader, optimizer, nn.CrossEntropyLoss(), device, args)
