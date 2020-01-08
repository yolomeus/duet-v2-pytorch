import os
from argparse import ArgumentParser

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_source import DuetHdf5Trainset
from duet_utils.io import load_config, DatasetConfigLoader
from duetv2_model import DuetV2

if __name__ == '__main__':
    parser = ArgumentParser(description='Train the DUET model.')
    parser.add_argument('-c', default='config', type=str, help='Path to the python config module (without .py)')
    parser.add_argument('-d',
                        default='FiQA',
                        choices=['FiQA', 'MSmarco', 'WikipassageQA', 'InsuranceQA'],
                        type=str,
                        help='Dataset to train on.')

    args = parser.parse_args()

    dataset_config = DatasetConfigLoader().get_dataset_config(args.c, args.d)
    train_config = load_config(args.c, 'TrainConfig')

    trainset = DuetHdf5Trainset(dataset_config.TRAIN_HDF5_PATH, 20, 200)
    data_loader = DataLoader(trainset, batch_size=train_config.BATCH_SIZE, shuffle=True)

    if torch.cuda.is_available():
        # cuda:0 will still use all GPUs
        device = torch.device('cuda:0')
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print('using {} device(s): "{}"'.format(torch.cuda.device_count(), dev_name))
    else:
        device = torch.device('cpu')

    model = DuetV2(75463, 128, out_features=1).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(20):
        model.train()

        for batch in tqdm(data_loader, desc='epoch {}'.format(epoch + 1)):
            batch = [x.to(device) for x in batch]
            query, pos_doc, neg_doc, pos_imat, neg_imat, labels = batch
            pos_out = model(query, pos_doc, pos_imat)
            neg_out = model(query, neg_doc, neg_imat)

            total_out = torch.cat([pos_out, neg_out], 1)

            batch_loss = loss(total_out, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(batch_loss)
