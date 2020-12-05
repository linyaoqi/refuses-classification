import warnings
warnings.filterwarnings('ignore')

import numpy as np
import re
import word2vec
import jieba

class w2v_wrapper:  ##加载词向量
     def __init__(self, file_path):
        self.model = word2vec.load(file_path)
        if 'unknown' not  in self.model.vocab_hash:  ##把词在链表中的位置存入到vocab_hash中
            unknown_vec = np.random.uniform(-0.1, 0.1, size=128)  ##均匀取128个值
            self.model.vocab_hash['unknown'] = len(self.model.vocab)
            self.model.vectors = np.row_stack((self.model.vectors, unknown_vec))  ##行合并
            
     def load_data_and_labels(filepath, max_size = -1):  ##加载数据和生成标签
        """Loads MR polarity data from files,  splits the data into words and generates labels.
        Returns split sentences and labels."""
        train_datas = []  ##加载数据
        with open(filepath,  'r',  encoding='utf-8', errors='ignore') as f:
            train_datas = f.readlines()
        one_hot_labels = []
        x_datas = []
        for line in train_datas:
            line = line.strip()
            parts = line.split('\t', 1)
            if(len(parts[1].strip()) == 0):
                continue
            words = jieba.cut(parts[1])
            x_datas.append(' '.join(words))
            one_hot = [0, 0, 0, 0] ##垃圾共4类
            one_hot[int(parts[0])] = 1
            one_hot_labels.append(one_hot)      
        print (' data size = ' , len(train_datas))  ##数据长度
        return [x_datas,  np.array(one_hot_labels)]  ##生成标签
            
def get_text_idx(text, vocab, max_document_length):  ##获取id
        text_array = np.zeros([len(text),  max_document_length], dtype=np.int32)  ##返回一个给定形状和类型的0数组
        for i, x in  enumerate(text):
            words = x.split(" ")
            for j,  w in enumerate(words):
                if w in vocab:
                    text_array[i,  j] = vocab[w]
                else :
                    text_array[i,  j] = vocab['unknown']
        return text_array

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(), !?\'\`]",  " ",  string)
    string = re.sub(r"\'s",  " \'s",  string)
    string = re.sub(r"\'ve",  " \'ve",  string)
    string = re.sub(r"n\'t",  " n\'t",  string)
    string = re.sub(r"\'re",  " \'re",  string)
    string = re.sub(r"\'d",  " \'d",  string)
    string = re.sub(r"\'ll",  " \'ll",  string)
    string = re.sub(r", ",  " ,  ",  string)
    string = re.sub(r"!",  " ! ",  string)
    string = re.sub(r"\(",  " \( ",  string)
    string = re.sub(r"\)",  " \) ",  string)
    string = re.sub(r"\?",  " \? ",  string)
    string = re.sub(r"\s{2, }",  " ",  string)
    return string.strip().lower()


def removezero( x,  y):
    nozero = np.nonzero(y)
    print('removezero', np.shape(nozero)[-1], len(y))

    if(np.shape(nozero)[-1] == len(y)):
        return np.array(x), np.array(y)

    y = np.array(y)[nozero]
    x = np.array(x)
    x = x[nozero]
    return x,  y


def read_file_lines(filename, from_size, line_num):
    i = 0
    text = []
    end_num = from_size + line_num
    for line in open(filename):
        if(i >= from_size):
            text.append(line.strip())

        i += 1
        if i >= end_num:
            return text

    return text

def batch_iter(data,  batch_size,  num_epochs,  shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size,  data_size)

            # print('epoch = %d, batch_num = %d, start = %d, end_idx = %d' % (epoch, batch_num, start_index, end_index))
            yield shuffled_data[start_index:end_index]

if __name__ == "__main__":
    x_text,  y = load_data_and_labels('./data/train_data.txt')
    print (len(x_text))
