from argparse import ArgumentParser

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model import DuetV2
from qa_utils.io import load_json_file, load_pkl_file


def main():
    ap = ArgumentParser(description='Train the DUET model.')
    ap.add_argument('TRAIN_DATA', help='Path to an hdf5 file containing the training data.')
    ap.add_argument('VAL_DATA', help='Path to an hdf5 file containing the validation data.')
    ap.add_argument('TEST_DATA', help='Path to an hdf5 file containing the test data.')
    ap.add_argument('VOCAB_FILE', help='JSON file containing the mapping from ids to words.')

    ap.add_argument('--max_epochs', type=int, default=20, help='Number of epochs')
    ap.add_argument('--gpus', type=int, nargs='+', help='GPU IDs to train on')
    ap.add_argument('--accumulate_grad_batches', type=int, default=1,
                    help='Update weights after this many batches')
    ap.add_argument('--logdir', default='logs', help='Working directory for checkpoints and logs')

    ap.add_argument('--val_patience', type=int, default=3, help='Validation patience')
    ap.add_argument('--random_seed', type=int, default=1579129142, help='Random seed')

    ap = DuetV2.add_model_specific_args(ap)
    args = ap.parse_args()

    seed_everything(args.random_seed)
    model = DuetV2(lr=args.learning_rate,
                   batch_size=args.batch_size,
                   idfs_file=args.IDF_FILE,
                   id_to_word_file=args.VOCAB_FILE,
                   glove_name=args.glove_name,
                   glove_cache=args.glove_cache,
                   glove_dim=args.glove_dim,
                   h_dim=args.hidden_dim,
                   max_q_len=args.max_q_len,
                   max_d_len=args.max_d_len,
                   dropout_rate=args.dropout,
                   train_file=args.TRAIN_DATA,
                   val_file=args.VAL_DATA,
                   test_file=args.TEST_DATA)

    early_stopping = EarlyStopping('val_mrr', mode='max', patience=args.val_patience)
    model_checkpoint = ModelCheckpoint(monitor='val_mrr', mode='max')
    # DDP seems to be buggy currently, so we use DP for now
    trainer = Trainer.from_argparse_args(args, distributed_backend='dp',
                                         default_root_dir=args.logdir,
                                         early_stop_callback=early_stopping,
                                         checkpoint_callback=model_checkpoint)
    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    main()
