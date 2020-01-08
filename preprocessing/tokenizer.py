import re

from qa_utils.text import Tokenizer


class DuetTokenizer(Tokenizer):
    """Tokenizer used for the original DUET implementation, taken from
    https://github.com/spacemanidol/MSMARCO/tree/master/Ranking/Baselines
    """

    def __init__(self):
        self.regex_drop_char = re.compile('[^a-z0-9\s]+')
        self.regex_multi_space = re.compile('\s+')

    def tokenize(self, s):
        return self.regex_multi_space.sub(' ', self.regex_drop_char.sub(' ', s.lower())).strip().split()
