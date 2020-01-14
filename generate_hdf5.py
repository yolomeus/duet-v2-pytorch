import os
from argparse import ArgumentParser

from preprocessing.hdf5_savers import DuetHhdf5Saver
from qa_utils.preprocessing.fiqa import FiQA
from qa_utils.preprocessing.msmarco import MSMARCO

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('DATA_DIR', type=str, help='Directory with the raw dataset files.')
    ap.add_argument('OUTPUT_DIR', type=str, help='Directory to store the generated files in.')
    ap.add_argument('DATA_SET', type=str, choices=['FIQA', 'MSMARCO'], help='The dataset that will be processed.')
    ap.add_argument('--vocab_size', type=int, default=80000,
                    help='Only use the n most frequent words for the vocabulary.')
    ap.add_argument('--num_neg_examples', type=int, default=1,
                    help='For each query sample this many negative documents.')
    ap.add_argument('--max_q_len', type=int, default=20, help='Maximum query length.')
    ap.add_argument('--max_d_len', type=int, default=200, help='Maximum document legth.')

    ap.add_argument('--no_train', default=False, action='store_true', help='Don\'t export the train set.')
    ap.add_argument('--no_dev', default=False, action='store_true', help='Don\'t export the dev set.')
    ap.add_argument('--no_test', default=False, action='store_true', help='Don\'t export the test set.')

    args = ap.parse_args()

    if args.DATA_SET == 'FIQA':
        args.FIQA_DIR = args.DATA_DIR
        args.SPLIT_FILE = os.path.join(args.DATA_DIR, 'fiqa_split.pkl')
        dataset = FiQA(args)
    elif args.DATA_SET == 'MSMARCO':
        args.MSM_DIR = args.DATA_DIR
        dataset = MSMARCO(args)
    else:
        raise NotImplementedError()

    train_path = None if args.no_train else os.path.join(args.OUTPUT_DIR, 'train.hdf5')
    dev_path = None if args.no_dev else os.path.join(args.OUTPUT_DIR, 'dev.hdf5')
    test_path = None if args.no_test else os.path.join(args.OUTPUT_DIR, 'test.hdf5')

    saver = DuetHhdf5Saver(dataset,
                           args.max_q_len,
                           args.max_d_len,
                           os.path.join(args.OUTPUT_DIR, 'vocabulary.pkl'),
                           os.path.join(args.OUTPUT_DIR, 'idfs.pkl'),
                           train_outfile=train_path,
                           dev_outfile=dev_path,
                           test_outfile=test_path,
                           max_vocab_size=args.vocab_size)

    saver.build_all()
