import json

from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import copy
import numpy as np
class ClassifyDataset(Dataset):
    def __init__(self, data_file_path, pad_token_idx=0):
            """
            构建分类数据集
            :param data_file_path: 数据文件路径
            """
            super(ClassifyDataset, self).__init__()
            self.PAD_IDX = pad_token_idx
            with open(data_file_path, 'rb') as f:
                self.datas = pickle.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        cat, y, x = self.datas[index]
        return copy.deepcopy(x), y, len(x)

    def collate_fn(self, batch):
        x, y, lengths = list(zip(*batch))
        max_length = max(lengths)  # 最大长度
        mask = np.zeros((len(x), max_length), dtype=np.float32)  # [N, T]
        for i in range(len(x)):
            x[i].extend([self.PAD_IDX] * (max_length - lengths[i]))  # 数据填充
            mask[i][:lengths[i]] = 1  # mask矩阵

        x = torch.tensor(x, dtype=torch.long)  # [N,T] N: batch_size, T: max_length
        y = torch.tensor(y, dtype=torch.long)  # [N,] N: batch_size
        lengths = torch.tensor(lengths, dtype=torch.long)  # [N,]
        mask = torch.from_numpy(mask)  # [N,T] N: batch_size, T: max_length
        return x, y, mask



def create_dataloader(file_path, batch_size, shuffle=False, num_workers=1, prefetch_factor=2):
    # 1.构建 Dataset 对象
    dataset = ClassifyDataset(file_path)
    # 2.构建 DataLoader 对象
    dataloader = DataLoader(dataset,  # 传入 Dataset 对象
                            batch_size=batch_size,  # 批处理大小
                            shuffle=shuffle,  # 是否打乱顺序，训练 True，评估或测试 False
                            num_workers=num_workers,  # 多线程读取的线程数， 一般默认值
                            prefetch_factor=prefetch_factor,  # 预取的批次数， 一般默认值
                            collate_fn=dataset.collate_fn  # 指定如何分批, True 为自定义分批方式 collate_fn 函数
                            )
    return dataloader



if __name__ == '__main__':
    tokens = json.load(open(r'./datasets/tokens.json', encoding='utf-8'))
    data = create_dataloader('./datasets/train.pkl', 4, True)
    for x_batch, y_batch, lengths_batch in data:
        # print(x_batch)
        print(type(x_batch))
        x_text = [''.join([tokens[token_id] for token_id in token_ids if token_id > 0]) for token_ids in x_batch.detach().numpy()]
        # x_batch.detach().numpy() 将tensor转换为numpy
        # 将token_id>0的token_id取出来
        print(x_text)
        print("="*20)
        print(y_batch)
        print("=" * 20)
        print(lengths_batch)
        break

