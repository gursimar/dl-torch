import io
import unicodedata
import unicodedata
import string
import re
import random
import torch
from torch import nn
from torch.autograd import Variable


# THE FUCTION IS CURRENTLY ONLY PRESENT IN MASTER
def pad_sequence(sequences, batch_first=False, padding_value=0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor


# Data acuitition
PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"PAD":0, "SOS":1, "EOS":2}
        self.word2count = {"PAD":1, "SOS":1, "EOS":1}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = io.open('data_att/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# Get cuda variables
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence, use_cuda):
    indexes = []
    indexes.append(SOS_token)
    indexes = indexes + indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair, input_lang, output_lang, use_cuda):
    input_variable = variableFromSentence(input_lang, pair[0], use_cuda)
    target_variable = variableFromSentence(output_lang, pair[1], use_cuda)
    return (input_variable, target_variable)


# Following class should be present

class DataLoader:
    def __init__(self, file_name, use_cuda):
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.PAD_token = PAD_token
        self.data_size = len(pairs)
        self.use_cuda = use_cuda

    def getBatch(self, indices):
        pairs = self.pairs
        input_lang = self.input_lang
        output_lang = self.output_lang
        use_cuda = self.use_cuda
        selected_pairs = [pairs[i] for i in indices]
        input_vars = []
        output_vars = []

        input_len = []
        target_len = []
        for sel_pair in selected_pairs:
            var_pair = variablesFromPair(sel_pair, input_lang, output_lang, use_cuda)
            input_vars.append(var_pair[0])
            output_vars.append(var_pair[1])
            input_len.append(var_pair[0].size(0))
            target_len.append(var_pair[1].size(0))
            
        input_len = Variable(torch.LongTensor(input_len))
        target_len = Variable(torch.LongTensor(target_len))
        if use_cuda:
            input_len = input_len.cuda()
            target_len = target_len.cuda()
        padded_inputs = pad_sequence(input_vars, True)
        padded_outputs = pad_sequence(output_vars, True)
        output = {}
        output['input'] = padded_inputs.squeeze()
        output['target'] = padded_outputs.squeeze()
        output['input_len'] = input_len
        output['target_len'] = target_len

        return output

# Unit test
if __name__ == "__main__": 
    dataloader = DataLoader('nothing', True)
    print(dataloader.getBatch([1,2]))
    print(dataloader.SOS_token)