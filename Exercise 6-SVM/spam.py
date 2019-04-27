import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from sklearn import svm
import re  # 电子邮件处理的正则表达式

# 这个英文算法似乎更符合作业里面所用的代码，与上面效果差不多
import nltk, nltk.stem.porter


def process_email(email):
    '''做除了Word Stemming和Removal of non-words的所有处理'''
    email = email.lower()
    email = re.sub('<[^<>]>', ' ',
                   email)  # 匹配<开头，然后所有不是< ,> 的内容，知道>结尾，相当于匹配<...>
    email = re.sub('(http|https)://[^\s]*', 'httpaddr',
                   email)  # 匹配//后面不是空白字符的内容，遇到空白字符则停止
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    email = re.sub('[\$]+', 'dollar', email)
    email = re.sub('[\d]+', 'number', email)
    return email


def email2TokenList(email):
    """预处理数据，返回一个干净的单词列表"""

    # I'll use the NLTK stemmer because it more accurately duplicates the
    # performance of the OCTAVE implementation in the assignment
    stemmer = nltk.stem.porter.PorterStemmer()

    email = process_email(email)

    # 将邮件分割为单个单词，re.split() 可以设置多种分隔符
    tokens = re.split(
        '[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    # 遍历每个分割出来的内容
    tokenlist = []
    for token in tokens:
        # 删除任何非字母数字的字符
        token = re.sub('[^a-zA-Z0-9]', '', token)
        # Use the Porter stemmer to 提取词根
        stemmed = stemmer.stem(token)
        # 去除空字符串‘’，里面不含任何字符
        if not len(token): continue
        tokenlist.append(stemmed)

    return tokenlist


def email2VocabIndices(email, vocab):
    """提取存在单词的索引"""
    token = email2TokenList(email)
    index = [i for i in range(len(vocab)) if vocab[i] in token]
    return index


def email_feature_vector(email):
    '''将email的单词转换为特征向量0/1'''
    df = pd.read_table('vocab.txt', names=['words'])
    vocab = df.values
    vector = np.zeros(len(vocab))
    vecab_indices = email2VocabIndices(email, vocab)
    for i in vecab_indices:
        vector[i] = 1
    return vector


def get_vocab_list():
    '''以字典形式获得词汇表'''
    vocab_dict = {}
    with open('vocab.txt') as f:  #打开txt格式的词汇表
        for line in f:
            (val, key) = line.split()  #读取每一行的键和值
            vocab_dict[int(val)] = key  #存放到字典中

    return vocab_dict


def main():

    # # ============================特征提取
    # with open('emailSample1.txt','r') as f:
    #     email=f.read()
    # feature=email_feature_vector(email)

    # #=============================计算现有的训练集合测试集精确度
    data = loadmat('spamTrain.mat')
    X = data['X']
    y = data['y'].flatten()

    # data2=loadmat('spamTest.mat')
    # Xtest,ytest=data2['Xtest'],data2['ytest']

    clf = svm.SVC(C=0.1, kernel='linear')
    clf.fit(X, y)

    # print(clf.score(X,y),clf.score(Xtest,ytest))

    #==============================通过训练好的分类器 打印权重最高的前15个词 邮件中出现这些词更容易是垃圾邮件

    vocab_list = get_vocab_list()  #得到词汇表 存在字典中
    indices = np.argsort(clf.coef_).flatten()[::-1]  #对权重序号进行从大到小排序 并返回

    for i in range(15):  #打印权重最大的前15个词 及其对应的权重
        print('{} ({:0.6f})'.format(vocab_list[indices[i]],
                                    clf.coef_.flatten()[indices[i]]))


if __name__ == "__main__":
    main()
