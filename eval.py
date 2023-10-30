"""
模型评估
"""

import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils import create_dataloader
from models.fc_model import FCTextClassifyModel
@torch.no_grad()  # 推理预测的时候，不需要计算梯度
def run(model_path, eval_path, batch_size, tokens, output_dir=Path("./output/01"), **kwargs):
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)  # 路径对象
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 2. 获取评估数据DataLoader对象
    eval_dataloader = create_dataloader(eval_path, batch_size=batch_size * 2)

    # 3. 模型恢复
    net = torch.load(model_path, map_location='cpu')
    net.eval()

    # 4. 遍历数据
    y_preds = []
    y_trues = []
    y_probs = []
    x_texts = []
    bar = tqdm(eval_dataloader)
    for x, y, mask in bar:
        scores = net(x, mask)  # 获取模型前向预测结果 [N,num_classes]
        probs = torch.softmax(scores, dim=1)  # 概率 [N,num_classes]
        y_pred = torch.argmax(scores, dim=1)  # [N,]
        # probs = torch.max(probs, dim=1)  # 预测为预测类别的概率
        probs = probs[:, 1]  # 预测为类别1的概率
        y_trues.append(y)
        y_preds.append(y_pred)
        y_probs.append(probs)
        x_text = [' '.join([tokens[token_id] for token_id in token_ids if token_id > 0]) for token_ids in x.detach().numpy()]
        x_texts.extend(x_text)
    y_preds = torch.concat(y_preds, dim=0).numpy()
    y_trues = torch.concat(y_trues, dim=0).numpy()
    accuracy = accuracy_score(y_trues, y_preds)
    print("Accuracy:", accuracy)

    # 构建DataFrame
    y_probs = torch.concat(y_probs, dim=0).numpy()
    df = pd.DataFrame([(y_trues == y_preds).astype('int'), y_trues, y_preds, y_probs, x_texts]).T
    df.columns = ['是否预测正确', '实际标签', '预测标签', '预测类别1概率', '原始文本']
    df.to_excel(str(eval_dir / "result.xlsx"), index=False)


def get_parser():
    import argparse

    # https://zhuanlan.zhihu.com/p/582298060?utm_id=0
    parser = argparse.ArgumentParser(description='入参')
    parser.add_argument('-model_path', type=str, default=r"./output/01/models/net_2.pkl", help='给定json token路径')
    parser.add_argument('-json_path', type=str, default=r"./datasets/tokens.json", help='给定json token路径')
    parser.add_argument('-train_path', type=str, default=r"./datasets/train.pkl", help='训练数据路径')
    parser.add_argument('-eval_path', type=str, default=r"./datasets/eval.pkl", help='测试数据路径')
    parser.add_argument('-batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('-embedding_dim', type=int, default=16, help='向量维度大小')
    parser.add_argument('-num_classes', type=int, default=2, help='分类类别数目')
    parser.add_argument('-total_epoch', type=int, default=10, help='总的训练epoch数量')
    parser.add_argument('-output_dir', type=str, default='./output/01', help='输出文件夹路径')
    return parser


def run_with_args():
    # 参数解析
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    print(os.path.abspath(args.json_path))
    print(os.path.abspath(args.model_path))
    tokens = json.load(open(args.json_path, "r", encoding="utf-8"))
    run(
        model_path=args.model_path,
        eval_path=args.eval_path,
        batch_size=args.batch_size,
        tokens=tokens,
        vocab_size=len(tokens),
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes,
        total_epoch=args.total_epoch,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    run_with_args()
