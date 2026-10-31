'''
案例：福彩双色球号码预测 —— RNN（循环神经网络）

思路：把「连续 10 期」当成一条长度为 10 的时间序列，每个时间步是一个 49 维向量

        时间步 t=0 : 第 1 期   49 维
        时间步 t=1 : 第 2 期   49 维
        ...
        时间步 t=9 : 第 10 期  49 维   ← 最新的一期

    网络从最旧的一期开始，一步一步往后"读"，每读一期就把信息压进隐状态 h，
    读完第 10 期时，h 里应当装着"这 10 期发生了什么"的摘要 —— 再用它预测第 11 期。

    ANN 一次看到 490 个数字，没有"先后顺序"的概念；
    CNN 用卷积核看局部窗口，也没有真正的记忆；
    RNN 是第一个带"时间顺序"的模型，这是三者最本质的区别。

步骤为标准训练脚本

1 准备训练、验证、测试集数据
2 模型搭建  RNN 层 + 全连接 + 两个输出头
3 训练代码（2244 骨架，和 ANN / CNN 完全一致）
4 绘制 loss 曲线
5 测试评估

# 和 _03_CNN.py 的差别只有两处：
#   1 数据直接喂 (B, 10, 49)，不需要 unsqueeze —— 循环网络天生就吃 3 维输入
#   2 模型第一层是 nn.RNN(input_size=49, hidden_size=64, batch_first=True)
#      batch_first=True 的意思是：输入和输出的第 0 维是 batch，形状是 (B, 序列长度, 特征数)
#      如果不加，PyTorch 默认按 (序列长度, B, 特征数) 解释，训练会悄悄算错
'''

# 导包
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# todo 0 让脚本在任何目录下运行都能导入项目根目录的模块
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import config
from _01_data.process_data import create_dataset_seq


# todo 0.1 随机种子
seed = config.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# todo 0.2 设备
if config.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备：{device}")
if device.type == "cuda":
    print(f"显卡：{torch.cuda.get_device_name(0)}")

# todo 0.3 路径与超参数
MODEL_PATH = config.model_path("rnn")     # _02_models/rnn_best.pth
LOSS_PATH = config.loss_path("rnn")       # _02_models/rnn_loss.png

BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LEARNING_RATE = config.learning_rate
WEIGHT_DECAY = config.weight_decay
LR_STEP_SIZE = config.lr_step_size
LR_GAMMA = config.lr_gamma
DROPOUT = config.dropout
PATIENCE = config.patience

RNN_HIDDEN = config.rnn_hidden_size       # 隐状态维度
RNN_LAYERS = config.rnn_num_layers        # 堆叠几层 RNN

FRONT_NUMBER = config.front_number
FRONT_COUNT = config.front_count
BACK_NUMBER = config.back_number


# todo 1 准备训练、验证、测试集数据
def create_dataset():
    """
        X         (N, 10, 49)   10 个时间步，每步 49 维
        Y_front   (N, 33)
        Y_back    (N,)

    返回：train_dataset, val_dataset, test_dataset, window_size, feature_number
    """
    # 返回的 dataset 都是 TensorDataset：把「特征张量 + 标签张量」按行打包，
    # dataset[0] 就是一条样本，例如 (x, y_front, y_back)；后面 DataLoader 再按批次取出来
    train_dataset, val_dataset, test_dataset, window_size, feature_number = create_dataset_seq()

    x0, yf0, yb0 = train_dataset[0]
    print(f"\n单条样本：X {tuple(x0.shape)}  "
          f"Y_front {tuple(yf0.shape)}  Y_back {tuple(yb0.shape)}（值 {yb0.item()}）")
    print(f"序列长度 {window_size}，每步特征 {feature_number}\n")

    return train_dataset, val_dataset, test_dataset, window_size, feature_number


# todo 2 模型搭建
class myModel(nn.Module):
    """
    结构：

        输入  (B, 10, 49)
          ↓ nn.RNN(input_size=49, hidden_size=64, num_layers=2, batch_first=True)
             out (B, 10, 64)   ← 每一个时间步的隐状态都输出
             h_n (2, B, 64)    ← 每一层最后一个时间步的隐状态
          ↓ 取 out[:, -1, :]   → (B, 64)   只保留"读完第 10 期"那一刻的状态
          ↓ Linear(64→32) + BN1d + LeakyReLU + Dropout
             (B, 32)
          ↓ 两个头
             front_head (B, 33) / back_head (B, 16)

    out[:, -1, :] 和 h_n[-1] 是同一个东西：
        out 是最后一层在"每个时间步"的输出；h_n 是"每一层"在最后一步的输出。
        取 out[:, -1, :] 的写法对双向 / 多层都通用，所以更常用。
    """

    def __init__(self, feature_number):

        super().__init__()

        # todo 2.1 循环层
        # batch_first=True 必须加：否则 PyTorch 会把第 0 维当成"时间步"，
        # 输入 (B, 10, 49) 会被理解成"10 个样本、B 个时间步、49 维"，形状对不上就报错了
        # dropout 只在 num_layers > 1 时生效；单层时传非 0 值 PyTorch 会警告
        self.rnn = nn.RNN(
            input_size=feature_number,     # 49
            hidden_size=RNN_HIDDEN,        # 64
            num_layers=RNN_LAYERS,         # 2
            batch_first=True,
            dropout=DROPOUT if RNN_LAYERS > 1 else 0.0,
        )

        # todo 2.2 全连接 + 两个输出头
        self.fc = nn.Sequential(
            nn.Linear(RNN_HIDDEN, RNN_HIDDEN // 2),
            nn.BatchNorm1d(RNN_HIDDEN // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(DROPOUT),
        )
        self.front_head = nn.Linear(RNN_HIDDEN // 2, FRONT_NUMBER)
        self.back_head = nn.Linear(RNN_HIDDEN // 2, BACK_NUMBER)

        # todo 2.3 权重初始化
        # RNN 自己的权重用 PyTorch 默认的均匀分布初始化（-1/√H ~ 1/√H），不改；
        # 只把后面全连接层按 kaiming 初始化，和 ANN / CNN 保持一致
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out, h_n = self.rnn(x)        # out: (B, 10, 64)
        last = out[:, -1, :]          # (B, 64)  最后一个时间步
        z = self.fc(last)             # (B, 32)
        return self.front_head(z), self.back_head(z)


# todo 3.7 的辅助函数：跑一遍验证集，返回平均 loss
# 单独抽成函数，是为了让下面的 train() 只保留主流程（2244 里的两组循环、四个动作），读起来更清楚
def evaluate(model, loader, criterion_front, criterion_back):
    model.eval()
    total_loss = 0.0
    total = 0

    # torch.no_grad()：关闭梯度计算。验证和推理都不需要反向传播，
    # 关掉可以省显存、也跑得更快（不关的话 PyTorch 会一直记着计算图）
    with torch.no_grad():
        for x, y_front, y_back in loader:
            # 循环网络直接吃 (B, 10, 49)，不需要 unsqueeze
            x = x.to(device)
            y_front = y_front.to(device)
            y_back = y_back.to(device)

            pred_front, pred_back = model(x)
            loss = criterion_front(pred_front, y_front) + criterion_back(pred_back, y_back)

            # loss.item()：把只有一个元素的张量取成 Python 浮点数。
            # 如果直接累加张量，整个计算图会一直被引用着，越算越占内存
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

    return total_loss / total


# todo 3 训练代码
def train(train_dataset, val_dataset, feature_number):

    # todo 3.1 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # todo 3.2 定义模型、损失函数、优化器、学习率调度器
    model = myModel(feature_number).to(device)

    criterion_front = nn.BCEWithLogitsLoss()
    criterion_back = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    loss_epoch_train, loss_epoch_valid = [], []
    best_val_loss = float('inf')
    wait = 0

    print("=" * 78)
    print(f"开始训练：{EPOCHS} 轮  批次 {BATCH_SIZE}  学习率 {LEARNING_RATE}  "
          f"参数量 {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 78)

    # todo 3.3 开启对轮数的循环
    for epoch in range(EPOCHS):

        start = time.time()
        total_loss = 0.0
        seen = 0

        model.train()

        # todo 3.4 开启对批次的循环
        for x, y_front, y_back in train_loader:

            x = x.to(device)
            y_front = y_front.to(device)
            y_back = y_back.to(device)

            pred_front, pred_back = model(x)

            loss = criterion_front(pred_front, y_front) + criterion_back(pred_back, y_back)

            # 梯度是累加的：每批开始前必须清零，
            # 否则这一批算出来的梯度会叠在上一批的梯度上，参数就更新错了
            optimizer.zero_grad()
            loss.backward()
            # 循环网络容易出现梯度爆炸，裁剪一下更稳（这一步可选，但建议留着）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            seen += x.size(0)

        # todo 3.5 记录当前轮的平均训练 loss
        avg_train_loss = total_loss / seen
        loss_epoch_train.append(avg_train_loss)

        # todo 3.6 学习率调整
        scheduler_lr.step()

        # todo 3.7 验证
        avg_val_loss = evaluate(model, val_loader, criterion_front, criterion_back)
        loss_epoch_valid.append(avg_val_loss)

        # todo 3.8 保存最佳模型
        flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # state_dict 是「参数名 → 张量」的字典：只存参数，不存模型结构。
            # 所以以后加载时，必须先 myModel(...) 建一个同样结构的空模型，再把参数填进去
            torch.save(model.state_dict(), MODEL_PATH)
            wait = 0
            flag = "  ← 保存最佳模型"
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"Epoch [{epoch + 1:>3}/{EPOCHS}]  "
                  f"训练 loss {avg_train_loss:.6f}  验证 loss {avg_val_loss:.6f}  "
                  f"学习率 {scheduler_lr.get_last_lr()[0]:.2e}  "
                  f"用时 {time.time() - start:.2f}s{flag}")

        # todo 3.9 早停
        if wait >= PATIENCE:
            print(f"\n早停：验证 loss 已连续 {PATIENCE} 轮没有下降，在第 {epoch + 1} 轮结束训练")
            break

    print(f"\n训练结束，最佳验证 loss：{best_val_loss:.6f}")
    print(f"模型已保存：{MODEL_PATH}")

    return loss_epoch_train, loss_epoch_valid


# todo 4 绘制 loss 曲线
def plot_loss(loss_epoch_train, loss_epoch_valid):

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_epoch_train) + 1), loss_epoch_train, color='red', label='Train loss')
    plt.plot(range(1, len(loss_epoch_valid) + 1), loss_epoch_valid, color='blue', label='Val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('RNN loss curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(LOSS_PATH, dpi=120)
    plt.close()

    print(f"loss 曲线已保存：{LOSS_PATH}")


# todo 5 测试评估
def test_fn(test_dataset, feature_number):

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # todo 5.1 定义模型，加载最优模型参数
    model = myModel(feature_number).to(device)
    # 两步：torch.load 从硬盘读出参数字典；load_state_dict 按参数名逐一填回模型
    # map_location=device 让「在 GPU 上训练的模型」也能在 CPU 上加载，反之亦然
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # todo 5.2 推理测试集，统计命中情况
    total = 0
    front_hit_sum = 0
    back_correct = 0
    hit_dist = {i: 0 for i in range(FRONT_COUNT + 1)}
    examples = []

    # 推理同样不需要梯度
    with torch.no_grad():
        for x, y_front, y_back in test_loader:

            x = x.to(device)
            y_front = y_front.to(device)
            y_back = y_back.to(device)

            pred_front, pred_back = model(x)

            front_pred = torch.topk(pred_front, k=FRONT_COUNT, dim=1).indices
            back_pred = torch.argmax(pred_back, dim=1)

            front_hit = y_front.gather(1, front_pred).sum(dim=1).long()
            front_hit_sum += front_hit.sum().item()
            back_correct += (back_pred == y_back).sum().item()

            for hit in front_hit.tolist():
                hit_dist[hit] += 1

            if len(examples) < 3:
                for i in range(min(3 - len(examples), x.size(0))):
                    examples.append((
                        [int(v) + 1 for v in front_pred[i].tolist()],
                        int(back_pred[i].item()) + 1,
                        [j + 1 for j in range(FRONT_NUMBER) if y_front[i][j] > 0.5],
                        int(y_back[i].item()) + 1,
                    ))

            total += x.size(0)

    avg_front_hit = front_hit_sum / total
    back_rate = back_correct / total
    random_front_hit = FRONT_COUNT * FRONT_COUNT / FRONT_NUMBER
    random_back_rate = 1 / BACK_NUMBER

    print("\n" + "=" * 78)
    print("测试集评估")
    print("=" * 78)
    print(f"  测试样本数        {total}")
    print(f"  前区平均命中      {avg_front_hit:.4f} / {FRONT_COUNT}"
          f"     （随机基线 {random_front_hit:.4f}）")
    print(f"  后区命中率        {back_rate:.2%}"
          f"             （随机基线 {random_back_rate:.2%}）")

    print(f"\n  前区命中数分布：")
    for hit, count in hit_dist.items():
        print(f"    中 {hit} 个：{count:>4} 条  ({count / total:>6.2%})")

    print(f"\n  预测 vs 实际（前 3 条）：")
    for i, (pf, pb, af, ab) in enumerate(examples, 1):
        hit = len(set(pf) & set(af))
        print(f"    第{i}条  预测 {' '.join(f'{n:02d}' for n in sorted(pf))} + {pb:02d}"
              f"   实际 {' '.join(f'{n:02d}' for n in af)} + {ab:02d}"
              f"   前区中 {hit} 个")

    print("\n  提示：双色球每期独立随机，指标贴着随机基线是正常结果。")


# todo 6 主程序
def main():
    train_dataset, val_dataset, test_dataset, window_size, feature_number = create_dataset()

    loss_epoch_train, loss_epoch_valid = train(train_dataset, val_dataset, feature_number)

    plot_loss(loss_epoch_train, loss_epoch_valid)

    test_fn(test_dataset, feature_number)


if __name__ == '__main__':
    main()
