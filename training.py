import sys
import os

# 防止import导入包异常的情况
# sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
# print(sys.path)

import json
from pathlib import Path
from tqdm import tqdm
import argparse  # 命令行参数解析库

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


from utils import create_dataloader
from models.fc_model import FCTextClassifyModel


def run(train_path, eval_path,
        batch_size=32, vocab_size=17025, embedding_dim=16, num_classes=2, total_epoch=100,
        output_dir=Path("./output/01")):

    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)  # 路径对象
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "logs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)


    # 1. 获取训练数据 DataLoader 对象
    train_dataloader = create_dataloader(train_path, batch_size=batch_size, shuffle=True)
    # 2. 获取测试数据 DataLoader 对象
    eval_dataloader = create_dataloader(eval_path, batch_size=batch_size * 2)


    # 3. 模型创建
    net = FCTextClassifyModel(vocab_size, embedding_dim, num_classes)
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 优化器
    writer = SummaryWriter(log_dir=str(summary_dir))
    # tensorboard 日志记录， 基于 TensorFlow 的 tensorboard 库
    # 如果运行报错，可以尝试安装 pip install tensorflow/ pip install tensorboard==2.11.2
    # 查看日志 - 命令行
    # - tensorboard --logdir E:\OneDrive\DeepBlue\Project\1_text_classify\output\01\logs
    # - （路径中最好不要有中文/空格）
    # 也可以使用 tensorboardX 库，基于 PyTorch 的 tensorboard 库
    # 如果运行报错，可以尝试安装 pip install tensorboardX

    # 当模型文件存在的时候，需要进行参数恢复
    model_file_names = os.listdir(str(models_dir))
    if len(model_file_names) > 0:
        model_file_names.sort(key=lambda x: int(x.split('_', maxsplit=2)[1].split('.')[0]))
        model_file_names = model_file_names[-1]
        ori_net = torch.load(str(models_dir / model_file_names), map_location=torch.device('cpu'))
        # 恢复模型参数
        # ori_net.state_dict() 返回的是一个dict字典对象，key是参数名，value是参数值
        # 底层是基于key进行参数匹配，然后进行恢复
        # 部分恢复：strict=False
        net.load_state_dict(ori_net.state_dict(),  strict=True)
        print('load model from {}'.format(str(models_dir / model_file_names)))

    # 4. 数据迭代训练
    for epoch in range(total_epoch):
        # 4.1 模型训练
        net.train()
        bar = tqdm(train_dataloader)
        for x, y, mask in bar:
            scores = net(x, mask)  # 前向传播 [batch_size, num_classes]
            loss = loss_fn(scores, y)  # 计算损失
            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 反向传播
            optimizer.step()        # 更新参数
            bar.set_postfix(epoch=epoch, train_loss=loss.item())  # 进度条提示
            writer.add_scalar('train_loss', loss.item(), global_step=epoch)  # tensorboard 日志记录
            writer.add_graph(net, torch.randint(vocab_size, size=(4, 8)))  # tensorboard 日志记录（模型结构图）

        # 4.2 模型评估
        with torch.no_grad():
            net.eval()
            y_preds = []
            y_trues = []
            bar = tqdm(eval_dataloader)
            for x, y, mask in bar:
                scores = net(x, mask)  # 前向传播 [batch_size, num_classes]
                y_pred = torch.argmax(scores, dim=1)  # [batch_size]
                y_trues.append(y)
                y_preds.append(y_pred)
            y_trues = torch.concat(y_trues, dim=0).numpy()  # [batch_size * len(eval_dataloader)]
            y_preds = torch.concat(y_preds, dim=0).numpy()  # [batch_size * len(eval_dataloader)]
            acc = accuracy_score(y_trues, y_preds)
            precision = precision_score(y_trues, y_preds)
            recall = recall_score(y_trues, y_preds)
            writer.add_scalar('eval_acc', acc, global_step=epoch)
            writer.add_scalar('precision', precision, global_step=epoch)
            writer.add_scalar('recall', recall, global_step=epoch)
            print('epoch: {}, acc: {}, precision: {}, recall: {}'.format(epoch, acc, precision, recall))


        # 模型阶段持久化
        if epoch % 1 == 0:
            torch.save(net, str(models_dir / f'net_{epoch}.pkl'))

    # 关闭 tensorboard 日志记录
    writer.close()

def get_parser():
    import argparse

    # https://zhuanlan.zhihu.com/p/582298060?utm_id=0
    parser = argparse.ArgumentParser(description='入参')
    parser.add_argument('-json_path', type=str, default=r"./datasets/tokens.json", help='给定json token路径')
    parser.add_argument('-train_path', type=str, default=r"./datasets/train.pkl", help='训练数据路径')
    parser.add_argument('-eval_path', type=str, default=r"./datasets/eval.pkl", help='测试数据路径')
    parser.add_argument('-batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('-embedding_dim', type=int, default=16, help='向量维度大小')
    parser.add_argument('-num_classes', type=int, default=2, help='分类类别数目')
    parser.add_argument('-total_epoch', type=int, default=3, help='总的训练epoch数量')
    parser.add_argument('-output_dir', type=str, default='./output/01', help='输出文件夹路径')
    return parser



if __name__ == '__main__':
    # 参数解析
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    tokens = json.load(open(args.json_path, "r", encoding="utf-8"))
    print('vocab_size', len(tokens))
    print(os.path.abspath(args.json_path))

    run(train_path=args.train_path,
        eval_path=args.eval_path,
        batch_size=args.batch_size,
        vocab_size=len(tokens),
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes,
        total_epoch=args.total_epoch,
        output_dir=args.output_dir)
