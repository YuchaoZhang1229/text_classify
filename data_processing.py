# -*- coding: utf-8 -*-
"""
数据处理模块
"""
import re  # 正则表达式
import pickle
import json
from pathlib import Path
import pandas as pd
import jieba
from tqdm import tqdm
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
data_file_path = r"./datasets/online_shopping_10_cats.csv"  # 数据集路径

def split_words(doc):
    words = jieba.lcut(doc)
    # 可以通过 token2count.json 查看分词效果
    return words

def stage1(data_file_path):
    """
    加载并预处理数据
    第一步：读取数据
    第二步：分词
    :param data_file_path:
    :return:
    """
    cats = set()
    labels = set()
    token2count = {}
    # 第一步：读取数据
    df = pd.read_csv(data_file_path)
    split_datasets = []
    print(df.head(3))
    # 第二步：分词
    for cat, label, review in tqdm(df.values):
        review = str(review).strip('\ufeff').strip()  # 去除文本中的特殊字符和空格
        review_token = split_words(review)

        cats.add(cat)
        labels.add(label)

        for token in review_token:
            token2count[token] = token2count.get(token, 0) + 1  # 统计词频

        split_datasets.append((cat, label, ' '.join(review_token)))

    # 数据保存
    output_dir = Path('./datasets')
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存分词后的数据
    with open(str(output_dir / 'cats.json'), 'w', encoding='utf-8') as f:
        json.dump(list(cats), f, ensure_ascii=False)

    with open(str(output_dir / 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(list(labels), f, ensure_ascii=False)

    token2count = list(token2count.items()) # 转换为列表, 便于排序
    token2count.sort(key=lambda x: x[1], reverse=True)  # 按照词频排序

    with open(str(output_dir / 'token2count.json'), 'w', encoding='utf-8') as f:
        json.dump(token2count, f, ensure_ascii=False, sort_keys=True, indent=2)

    # 最小词频阈值为5
    tokens = [token for token, count in token2count if count >= 5]
    # 特殊处理：数字，标点符号，符号，未知词，填充词
    for token in ['<NUM>', '<PUN>', '<SYMBOLS>', '<UNK>', '<PAD>']:
        tokens.insert(0, token)

    with open(str(output_dir / 'tokens.json'), 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False, sort_keys=True, indent=2)

    df = pd.DataFrame(split_datasets, columns=['cat', 'label', 'review'])
    df.to_csv(str(output_dir/'split_datasets.csv'), index=False)


re_num = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')  # 匹配数字

def is_number(token):
    return bool(re_num.match(token))

def is_punctuation(token):
    pattern = r'^\W$'
    if re.match(pattern, token):
        return True
    else:
        return False

def is_symbol(token):
    symbol = ['~', '!', '@', '#', '$', '%']
    if token in symbol:
        return True
    else:
        return False

def stage2():
    """
    第三步：将分词后的数据转换为id
    第四步: 划分训练集和测试集
    :return:
    """
    tokens = json.load(open('./datasets/tokens.json', encoding='utf-8'))
    token2id = dict(zip(tokens, range(len(tokens))))  # 词到id的映射, 从0开始

    cats = json.load(open('./datasets/cats.json', encoding='utf-8'))
    cats2id = dict(zip(cats, range(len(cats))))  # 词到id的映射, 从0开始

    labels = json.load(open('./datasets/labels.json', encoding='utf-8'))
    labels2id = dict(zip(labels, range(len(labels))))  # 词到id的映射, 从0开始

    unk_token_id = token2id['<UNK>']  # 未知词的id

    in_file = r'./datasets/split_datasets.csv'  # 分词后的数据集
    df = pd.read_csv(in_file, sep=",")  # 读取数据, 以逗号分隔
    print(df.head(3))


    datas = []
    for cat, label, review in tqdm(df.values):
        tokens = str(review).split(' ')  # 以空格分隔
        tokenids = []
        for token in tokens:
            if is_number(token):
                tokenids.append(token2id['<NUM>'])  # 数字的id
            elif is_punctuation(token):
                tokenids.append(token2id['<PUN>'])  # 标点符号的id
            elif is_symbol(token):
                tokenids.append(token2id['<SYMBOLS>'])
            else:
                tokenids.append(token2id.get(token, unk_token_id))  # 未知词的id

        datas.append((cats2id[cat], labels2id[label], tokenids)) # 保存为元组 (cat, label, tokenids)

    # 划分训练集和测试集
    train_datas, eval_datas = train_test_split(datas, test_size=0.2, random_state=42)
    print(f"训练集大小：{len(train_datas)}")
    print(f"测试集大小：{len(eval_datas)}")

    # 保存数据
    with open('./datasets/train.pkl', 'wb') as f:
        pickle.dump(train_datas, f)
    with open('./datasets/eval.pkl', 'wb') as f:
        pickle.dump(eval_datas, f)

# main
if __name__ == '__main__':
    stage2()