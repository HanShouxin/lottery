'''
案例：福彩双色球号码预测 —— GRU（门控循环单元）

为什么要有 GRU：
    LSTM 效果好但参数多（h 和 c 两条状态、三个门、4 组权重）。
    GRU 是它的精简版，只保留 h 一条状态、两个门：
        更新门 z : 决定"旧记忆保留多少、新信息写入多少"（把 LSTM 的遗忘门和输入门合并成一个）
        重置门 r : 决定"计算新候选记忆时，要看多少旧记忆"
    结果是参数量比 LSTM 少约 1/4，多数任务上效果相当，训练也更快。

    三者的关系可以这样记：
        RNN  → 只有一个隐状态，直接覆盖
        GRU  → 加了两个门，控制覆盖的比例
        LSTM → 加了两条状态 + 三个门，控制得更细

步骤为标准训练脚本

1 准备训练、验证、测试集数据
2 模型搭建  GRU 层 + 全连接 + 两个输出头
3 训练代码（2244 骨架，和 ANN / CNN / RNN / LSTM 完全一致）
4 绘制 loss 曲线
5 测试评估

# 和 _05_LSTM.py 的差别只有两处：
#   1 nn.LSTM → nn.GRU
#   2 前向传播少一层括号：out, h_n = self.gru(x)
#      （LSTM 是 out, (h_n, c_n) = self.lstm(x)，因为它有两条状态）
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
MODEL_PATH = config.model_path("gru")     # _02_models/gru_best.pth
LOSS_PATH = config.loss_path("gru")       # _02_models/gru_loss.png

BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LEARNING_RATE = config.learning_rate
WEIGHT_DECAY = config.weight_decay
LR_STEP_SIZE = config.lr_step_size
LR_GAMMA = config.lr_gamma
DROPOUT = config.dropout
PATIENCE = config.patience

GRU_HIDDEN = config.rnn_hidden_size       # 隐状态维度
GRU_LAYERS = config.rnn_num_layers        # 堆叠几层 GRU

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
          ↓ nn.GRU(input_size=49, hidden_size=64, num_layers=2, batch_first=True)
             out (B, 10, 64)   每个时间步的输出
             h_n (2, B, 64)    各层最后一步的隐状态
          ↓ 取 out[:, -1, :] → (B, 64)
          ↓ Linear(64→32) + BN1d + LeakyReLU + Dropout
          ↓ 两个头  front_head (B, 33) / back_head (B, 16)

    参数量对比（input=49, hidden=64, layers=2）：
        RNN  ≈ 3 × (49×64 + 64×64 + 64)      ≈ 22k
        GRU  ≈ 3 × (3 × 64×64) 的一部分      → 比 RNN 多
        LSTM ≈ 4 组权重，比 GRU 再多约 1/4
    跑起来看"参数量"那一行最直观。
    """

    def __init__(self, feature_number):

        super().__init__()

        # todo 2.1 循环层
        self.gru = nn.GRU(
            input_size=feature_number,     # 49
            hidden_size=GRU_HIDDEN,        # 64
            num_layers=GRU_LAYERS,         # 2
            batch_first=True,
            dropout=DROPOUT if GRU_LAYERS > 1 else 0.0,
        )

        # todo 2.2 全连接 + 两个输出头
        self.fc = nn.Sequential(
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN // 2),
            nn.BatchNorm1d(GRU_HIDDEN // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(DROPOUT),
        )
        self.front_head = nn.Linear(GRU_HIDDEN // 2, FRONT_NUMBER)
        self.back_head = nn.Linear(GRU_HIDDEN // 2, BACK_NUMBER)

        # todo 2.3 权重初始化（只处理全连接层，GRU 内部权重保留 PyTorch 默认初始化）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # GRU 和 RNN 一样只返回两条：输出 + 最后一步的隐状态
        out, h_n = self.gru(x)        # out: (B, 10, 64)
        last = out[:, -1, :]          # (B, 64)
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
            # 梯度裁剪，和 RNN / LSTM 保持一致
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
    plt.title('GRU loss curve')
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
