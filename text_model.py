"""Class for text data."""
import string
import numpy as np
import torch
import json


class SimpleVocab(object):

    def __init__(self):
        super(SimpleVocab, self).__init__()
        self.word2id = {}
        self.wordcount = {}
        self.word2id['<UNK>'] = 0
        self.word2id['<AND>'] = 1
        self.word2id['<BOS>'] = 2
        self.word2id['<EOS>'] = 3
        self.wordcount['<UNK>'] = 9e9
        self.wordcount['<AND>'] = 9e9
        self.wordcount['<BOS>'] = 9e9
        self.wordcount['<EOS>'] = 9e9

    def tokenize_text(self,text):   # 把字符串去掉符号变成单词的list
        text = text.encode('ascii', 'ignore').decode('ascii')
        trans=str.maketrans({key: None for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        return tokens
    
    def add_text_to_vocab(self,text):   # 每个单词建立word2id 以及 wordcount 的字典
        tokens = self.tokenize_text(text)
        for token in tokens:
            if not token in self.word2id:
                self.word2id[token] = len(self.word2id)
                self.wordcount[token] = 0
            self.wordcount[token] += 1

    def threshold_rare_words(self, wordcount_threshold=3):  # 去掉出现的单词数量小于5的词
        for w in self.word2id:
            if self.wordcount[w] < wordcount_threshold:
                self.word2id[w] = 0

    def encode_text(self, text):                #把句子转变成数字
        tokens = self.tokenize_text(text)
        x = [self.word2id.get(t, 0) for t in tokens]           # dict.get(key, default=None)
        return x
    
    def get_size(self):                #单词数量
        return len(self.word2id)



class TextLSTMModel(torch.nn.Module):

    def __init__(self,
                 texts_to_build_vocab = None,
                 word_embed_dim = 512,
                 lstm_hidden_dim = 512):

        super(TextLSTMModel, self).__init__()

        self.vocab = SimpleVocab()
        if texts_to_build_vocab != None:
            for text in texts_to_build_vocab:
                self.vocab.add_text_to_vocab(text)
        else:
            vocab_data = json.load(open("simplevocab.json"))
            self.vocab.word2id = vocab_data['word2id']
            self.vocab.wordcount = vocab_data['wordcount']

        vocab_size = self.vocab.get_size()
        self.word_embed_dim = word_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.embedding_layer = torch.nn.Embedding(vocab_size, word_embed_dim)
        self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim)


    def forward(self, x):
        """ input x: list of strings"""
        if type(x) is list:
            if type(x[0]) is str or type(x[0]) is unicode:
                x = [self.vocab.encode_text(text) for text in x]

        assert type(x) is list
        assert type(x[0]) is list
        assert type(x[0][0]) is int
        return self.forward_encoded_texts(x)

    def forward_encoded_texts(self, texts):
        # to tensor
        lengths = [len(t) for t in texts]
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for i in range(len(texts)):
            itexts[:lengths[i], i] = torch.tensor(texts[i])

        # embed words
        itexts = torch.autograd.Variable(itexts).cuda()     # shape(length,batch)
        etexts = self.embedding_layer(itexts)               # shape(length,batch,dim)         

        # lstm
        lstm_output, _ = self.forward_lstm_(etexts)      #lstm_output shape(length,batch,hidden_num*directions)
        return lstm_output.permute(1,0,2).contiguous()


    def forward_lstm_(self, etexts):
        batch_size = etexts.shape[1]
        first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
                        torch.zeros(1, batch_size, self.lstm_hidden_dim))
        first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
        #first_hidden = (first_hidden[0], first_hidden[1])
        lstm_output, last_hidden = self.lstm(etexts, first_hidden)
        return lstm_output, last_hidden


        

