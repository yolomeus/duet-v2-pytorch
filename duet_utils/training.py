import csv
import os

import torch
from tqdm import tqdm

from duet_utils.io import batch_to_device
from qa_utils.misc import Logger


def train_model_pairwise(model, train_dl, optimizer, criterion, device, args):
    """Train a model using a pairwise loss function. Save the model after each epoch and log the loss in
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
    with open(args_file, 'w') as fp:
        writer = csv.writer(fp)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])

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

        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch + 1))
        print('saving {}...'.format(fname))
        torch.save(state, fname)
