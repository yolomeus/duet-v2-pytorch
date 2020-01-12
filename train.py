import csv
import os
from argparse import ArgumentParser

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_source import DuetHdf5Trainset
from duetv2_model import DuetV2
from qa_utils.io import batch_to_device, get_cuda_device, load_pkl_file
from qa_utils.misc import Logger


def train_model_pairwise_ce(model, train_dl, optimizer, device, args):
    """Train a model using pairwise CrossentropyLoss. Save the model after each epoch and log the loss in
        a file.

        Arguments:
            model {torch.nn.Module} -- The model to train
            train_dl {torch.utils.data.DataLoader} -- Train dataloader
            optimizer {torch.optim.Optimizer} -- Optimizer
            args {argparse.Namespace} -- All command line arguments
            device {torch.device} -- Device to train on
        """
    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_file, ['epoch', 'loss'])

    # save all args in a file
    args_file = os.path.join(args.working_dir, 'args.csv')
    print('writing {}...'.format(args_file))
    with open(args_file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])

    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0
        optimizer.zero_grad()
        for i, (b_pos, b_neg, b_y) in enumerate(tqdm(train_dl, desc='epoch {}'.format(epoch + 1))):
            pos_out = model(*batch_to_device(b_pos, device))
            neg_out = model(*batch_to_device(b_neg, device))
            out = torch.cat([pos_out, neg_out], 1)

            loss = criterion(out, b_y.to(device)) / args.accumulate_batches
            loss.backward()
            if (i + 1) % args.accumulate_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss.item()

        epoch_loss = loss_sum / len(train_dl)
        print('epoch {} -- loss: {}'.format(epoch + 1, epoch_loss))
        logger.log([epoch + 1, epoch_loss])

        state = {'epoch': epoch + 1, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch + 1))
        print('saving {}...'.format(fname))
        torch.save(state, fname)


def main():
    ap = ArgumentParser(description='Train the DUET model.')
    ap.add_argument('TRAIN_DATA', help='Path to an hdf5 file containing the training data.')
    ap.add_argument('VOCAB_FILE', help='Pickle file containing the mapping from ids to words.')
    ap.add_argument('IDF_FILE', help='Pickle file containing the mapping from ids to words.')

    ap.add_argument('--glove_name', default='840B', help='GloVe embedding name')
    ap.add_argument('--glove_cache', default='glove_cache', help='Glove cache directory.')
    ap.add_argument('--glove_dim', type=int, default=300, help='The dimensionality of the GloVe embeddings')

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
    ap.add_argument('--random_seed', type=int, default=38852956087345243, help='Random seed')

    args = ap.parse_args()

    torch.manual_seed(args.random_seed)

    idfs = load_pkl_file(args.IDF_FILE)
    trainset = DuetHdf5Trainset(args.TRAIN_DATA, args.max_q_len, args.max_d_len, idfs)
    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    device = get_cuda_device()
    id_to_word = load_pkl_file(args.VOCAB_FILE)
    model = DuetV2(id_to_word=id_to_word,
                   glove_name=args.glove_name,
                   glove_cache=args.glove_cache,
                   glove_dim=args.glove_dim,
                   h_dim=args.hidden_dim,
                   max_q_len=args.max_q_len,
                   max_d_len=args.max_d_len,
                   dropout_rate=args.dropout)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_model_pairwise_ce(model, train_dataloader, optimizer, device, args)


if __name__ == '__main__':
    main()
