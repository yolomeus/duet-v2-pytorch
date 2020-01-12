import argparse

import torch
from torch.utils.data import DataLoader

from data_source import DuetHdf5Testset

from duetv2_model import DuetV2
from qa_utils.evaluation import read_args, evaluate_all
from qa_utils.io import get_cuda_device, load_pkl_file

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('DEV_DATA', help='Dev data hdf5 filepath.')
    ap.add_argument('TEST_DATA', help='Test data hdf5 filepath.')
    ap.add_argument('WORKING_DIR', help='Working directory containing args.csv and a ckpt folder.')
    ap.add_argument('--mrr_k', type=int, default=10, help='Compute MRR@k')
    ap.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    args = ap.parse_args()

    train_args = read_args(args.WORKING_DIR)

    max_q_len = int(train_args['max_q_len'])
    max_d_len = int(train_args['max_d_len'])

    dev_set = DuetHdf5Testset(args.DEV_DATA, max_q_len, max_d_len)
    test_set = DuetHdf5Testset(args.TEST_DATA, max_q_len, max_d_len)

    dev_dl = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_dl = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    device = get_cuda_device()

    id_to_word = load_pkl_file(train_args['VOCAB_FILE'])
    model = DuetV2(id_to_word=id_to_word,
                   glove_name=train_args['glove_name'],
                   glove_cache=train_args['glove_cache'],
                   glove_dim=int(train_args['glove_dim']),
                   h_dim=int(train_args['hidden_dim']),
                   max_q_len=int(train_args['max_q_len']),
                   max_d_len=int(train_args['max_d_len']),
                   dropout_rate=float(train_args['dropout']))
    model.to(device)
    model = torch.nn.DataParallel(model)
    evaluate_all(model, args.WORKING_DIR, dev_dl, test_dl, args.mrr_k, device, multi_input_model=True)
