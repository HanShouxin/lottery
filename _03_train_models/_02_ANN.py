'''
案例：福彩双色球号码预测 —— ANN（全连接神经网络）

为什么用彩票做案例：
    双色球每期的开奖号码是独立随机事件，模型学不到真实规律，指标最终会贴近随机基线。
    这个项目的价值不在"预测准不准"，而在把「数据处理 → 建模 → 训练 → 评估 → 部署」这条链路完整走一遍。

步骤为标准训练脚本，摈弃教材 PPT 中的套路

要求  不需要默写，但是需要知道流程，看到代码能想到用哪个 API 或函数

1 准备训练、验证、测试集数据 生成数据集对象
   1.1  读入表格数据（由 _01_data/process_data.py 统一完成）
   1.2  分割特征列和标签列（前区 33 维多标签 + 后区 16 类单标签）
   1.3  数据集分割（按时间 8:1:1）
   1.4  创建训练集对象、验证集对象、测试集对象

2 模型搭建  三层全连接 + 两个输出头

3 训练代码
    3.1  创建数据加载器
    3.2  定义模型、优化器、学习率衰减、损失函数
    3.3  开启对轮数的循环
          模型调整为训练模式
          3.4 开启对批次的循环
                  正向传播 → 计算损失 → 梯度清零 → 反向传播 → 参数更新
          3.5 记录当前轮的平均训练 loss
          3.6 学习率调整
          3.7 模型切换为验证模式，跑一遍验证集
          3.8 保存模型参数
4 绘制 loss 下降曲线

5 测试评估

# 2244  一套完整训练代码包含的所有要素
# 第一个2： 创建数据集对象，以及 DataLoader
# 第二个2： 搭建2个循环。第一个是对数据集整体循环几轮（epoch）；第二个是对每个数据批次循环（step）
# 第一个4： 定义模型，定义损失函数，定义优化器，定义学习率
# 第二个4： （每个批次）梯度清零 zero_grad()、反向传播 backward()、优化器更新 step()
#          （每个轮次）更新学习率 scheduler.step()

# 和 example_code/ANN.py 的差别
#   1 数据不再在脚本里手写，统一调用 _01_data/process_data.py 的 create_dataset_ann()
#   2 标签有两套：前区用多标签（BCEWithLogitsLoss），后区用多分类（CrossEntropyLoss），所以模型有两个输出头
#   3 不做 StandardScaler 标准化：输入是 one-hot 的 0/1，标准化会破坏稀疏语义
#   4 按时间顺序切分数据集，不能 shuffle，否则验证集会混进"未来"的数据
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
# 因为 _03_train_models 是子目录，Python 默认找不到根目录的 config.py 和 _01_data/
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import config
from _01_data.process_data import create_dataset_ann


# todo 0.1 随机种子：让每次训练的结果可复现
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
# 注意：没 GPU 时 torch.cuda.get_device_name(0) 会直接报错，所以必须先判断
if device.type == "cuda":
    print(f"显卡：{torch.cuda.get_device_name(0)}")

# todo 0.3 路径与超参数：全部来自 .env，改参数不用动代码
MODEL_PATH = config.model_path("ann")     # _02_models/ann_best.pth
LOSS_PATH = config.loss_path("ann")       # _02_models/ann_loss.png

BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LEARNING_RATE = config.learning_rate
WEIGHT_DECAY = config.weight_decay
LR_STEP_SIZE = config.lr_step_size
LR_GAMMA = config.lr_gamma
HIDDEN_SIZE = config.hidden_size
DROPOUT = config.dropout
PATIENCE = config.patience

FRONT_NUMBER = config.front_number        # 33
FRONT_COUNT = config.front_count          # 6
BACK_NUMBER = config.back_number          # 16


# todo 1 准备训练、验证、测试集数据
def create_dataset():
    """
    数据全部由 _01_data/process_data.py 统一提供，本脚本不碰任何编码细节。

    拿到的数据：
        X         (N, 490)   连续 10 期 × 每期 49 维 one-hot，首尾相接
        Y_front   (N, 33)    前区多标签，一期 6 个 1
        Y_back    (N,)       后区类别下标 0~15，一期 1 个整数
    """
    # 返回的 dataset 都是 TensorDataset：把「特征张量 + 标签张量」按行打包，
    # dataset[0] 就是一条样本，例如 (x, y_front, y_back)；后面 DataLoader 再按批次取出来
    train_dataset, val_dataset, test_dataset, feature_number = create_dataset_ann()

    # 取一条样本看看形状，确认数据没串行
    x0, yf0, yb0 = train_dataset[0]
    print(f"\n单条样本：X {tuple(x0.shape)}  "
          f"Y_front {tuple(yf0.shape)}  Y_back {tuple(yb0.shape)}（值 {yb0.item()}）")
    print(f"输入特征数：{feature_number}\n")

    return train_dataset, val_dataset, test_dataset, feature_number


# todo 2 模型搭建
class myModel(nn.Module):
    """
    全连接网络：490 → 128 → 64 → 32，然后分两个头

        front_head: 32 → 33   前区每个号码一个得分（多标签，6 个位置可以同时为 1）
        back_head:  32 → 16   后区每个号码一个得分（16 分类，取最大的那个）

    每个隐藏层都是 Linear → BatchNorm1d → LeakyReLU → Dropout：
        BatchNorm  把每层输出拉回稳定的分布，训练更快、更不容易梯度消失
        LeakyReLU  负半轴保留一个很小的斜率，避免神经元"死掉"
        Dropout    训练时随机丢一部分神经元，抑制过拟合
    """

    def __init__(self, feature_number):
        super().__init__()

        self.net = nn.Sequential(
            # 第一层：490 → 128
            nn.Linear(feature_number, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE, eps=1e-5, momentum=0.1, affine=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(DROPOUT),

            # 第二层：128 → 64
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.BatchNorm1d(HIDDEN_SIZE // 2, eps=1e-5, momentum=0.1, affine=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(DROPOUT),

            # 第三层：64 → 32
            nn.Linear(HIDDEN_SIZE // 2, HIDDEN_SIZE // 4),
            nn.BatchNorm1d(HIDDEN_SIZE // 4, eps=1e-5, momentum=0.1, affine=True),
            nn.LeakyReLU(negative_slope=0.1),
        )

        # 两个输出头，共用前面提取的特征
        self.front_head = nn.Linear(HIDDEN_SIZE // 4, FRONT_NUMBER)
        self.back_head = nn.Linear(HIDDEN_SIZE // 4, BACK_NUMBER)

        # todo 遍历所有 Linear 层，做 kaiming 正态初始化（就地修改 weight）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.net(x)
        # 这里输出的是 logits（没有过 sigmoid / softmax），
        # 因为 BCEWithLogitsLoss 内部自带 sigmoid，CrossEntropyLoss 内部自带 softmax
        return self.front_head(x), self.back_head(x)


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

            # 注意这里要乘 batch 的大小：一个 batch 的 loss 是批内平均值，
            # 想还原成"总 loss"，就得乘这一批有多少条数据
            # loss.item()：把只有一个元素的张量取成 Python 浮点数。
            # 如果直接累加张量，整个计算图会一直被引用着，越算越占内存
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

    return total_loss / total


# todo 3 训练代码
def train(train_dataset, val_dataset, feature_number):

    # todo 3.1 创建数据加载器
    # drop_last=True：丢掉最后一个不满批次的数据。
    #   因为模型里有 BatchNorm1d，如果最后一批只剩 1 条数据，BN 没法算方差会直接报错。
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # todo 3.2 定义模型、损失函数、优化器、学习率调度器
    model = myModel(feature_number).to(device)

    # 前区：6 个号码各自独立判断"是 / 不是"，所以是多标签 + 二分类交叉熵
    criterion_front = nn.BCEWithLogitsLoss()
    # 后区：从 16 个号码里挑 1 个，是多分类，标签必须是类别下标
    criterion_back = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY
    )

    scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # 用于存放每一轮训练后在训练集和验证集上的平均 loss
    loss_epoch_train, loss_epoch_valid = [], []
    best_val_loss = float('inf')
    wait = 0          # 验证 loss 已经连续多少轮没下降了（早停计数）

    print("=" * 78)
    print(f"开始训练：{EPOCHS} 轮  批次 {BATCH_SIZE}  学习率 {LEARNING_RATE}  "
          f"参数量 {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 78)

    # todo 3.3 开启对轮数的循环
    for epoch in range(EPOCHS):

        start = time.time()
        total_loss = 0.0
        seen = 0

        # todo 模型调整为训练模式（开启 Dropout，BN 用当前批次自己的均值方差）
        model.train()

        # todo 3.4 开启对批次的循环
        for x, y_front, y_back in train_loader:

            x = x.to(device)
            y_front = y_front.to(device)
            y_back = y_back.to(device)

            # todo 正向传播得到预测值
            pred_front, pred_back = model(x)

            # todo 计算损失：两个头的损失相加，一次反向传播同时更新
            loss = criterion_front(pred_front, y_front) + criterion_back(pred_back, y_back)

            # todo 梯度清零、反向传播、参数更新
            # 梯度是累加的：每批开始前必须清零，
            # 否则这一批算出来的梯度会叠在上一批的梯度上，参数就更新错了
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # todo 当前轮 loss 累加
            total_loss += loss.item() * x.size(0)
            seen += x.size(0)

        # todo 3.5 记录当前轮的平均训练 loss
        avg_train_loss = total_loss / seen
        loss_epoch_train.append(avg_train_loss)

        # todo 3.6 学习率调整
        scheduler_lr.step()

        # todo 3.7 模型切换为验证模式，跑一遍验证集
        avg_val_loss = evaluate(model, val_loader, criterion_front, criterion_back)
        loss_epoch_valid.append(avg_val_loss)

        # todo 3.8 保存验证集上表现最好的模型参数
        flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # state_dict 是「参数名 → 张量」的字典：只存参数，不存模型结构。
            # 所以以后加载时，必须先 myModel(...) 建一个同样结构的空模型，再把参数填进去
            torch.save(model.state_dict(), MODEL_PATH)
            wait = 0
            flag = "  ← 保存最佳模型"
        else:
            # 验证 loss 没降：累计"耐心值"，连续 PATIENCE 轮都没降就提前结束
            wait += 1

        # 每轮都打印会刷屏，这里每 10 轮打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"Epoch [{epoch + 1:>3}/{EPOCHS}]  "
                  f"训练 loss {avg_train_loss:.6f}  验证 loss {avg_val_loss:.6f}  "
                  f"学习率 {scheduler_lr.get_last_lr()[0]:.2e}  "
                  f"用时 {time.time() - start:.2f}s{flag}")

        # todo 3.9 早停：验证 loss 长期不降，说明模型开始"死记硬背"训练集了
        if wait >= PATIENCE:
            print(f"\n早停：验证 loss 已连续 {PATIENCE} 轮没有下降，"
                  f"在第 {epoch + 1} 轮结束训练")
            break

    print(f"\n训练结束，最佳验证 loss：{best_val_loss:.6f}")
    print(f"模型已保存：{MODEL_PATH}")

    return loss_epoch_train, loss_epoch_valid


# todo 4 绘制训练集和验证集的 loss 下降曲线
def plot_loss(loss_epoch_train, loss_epoch_valid):

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_epoch_train) + 1), loss_epoch_train, color='red', label='Train loss')
    plt.plot(range(1, len(loss_epoch_valid) + 1), loss_epoch_valid, color='blue', label='Val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('ANN loss curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(LOSS_PATH, dpi=120)
    # 想看弹窗就把下面这行换成 plt.show()（后台运行时弹窗会卡住）
    plt.close()

    print(f"loss 曲线已保存：{LOSS_PATH}")


# todo 5 测试评估
def test_fn(test_dataset, feature_number):
    """
    双色球没法用"准确率"衡量：预测 6 个号，全中才有意义，绝大多数情况一个都不中。
    这里用更合理的两个指标：

        前区平均命中数   随机猜的期望值是 6 × 6/33 = 1.09 个
        后区命中率       随机猜的期望值是 1/16 = 6.25%

    模型的结果只会贴着这两条基线 —— 这就是双色球的本质，不是代码写错了。
    """
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # todo 5.1 定义模型，加载最优模型参数
    model = myModel(feature_number).to(device)
    # 两步：torch.load 从硬盘读出参数字典；load_state_dict 按参数名逐一填回模型
    # map_location=device 让「在 GPU 上训练的模型」也能在 CPU 上加载，反之亦然
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # todo 5.2 依次推理测试集，统计命中情况
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

            # 前区：取 33 个得分里最高的 6 个。topk 的下标天然不重复，正好是 6 个不同的号码
            front_pred = torch.topk(pred_front, k=FRONT_COUNT, dim=1).indices
            # 后区：取 16 个得分里最大的那个
            back_pred = torch.argmax(pred_back, dim=1)

            # 前区命中数：拿标签（one-hot）去查预测出来的 6 个位置，是 1 就说明中了
            front_hit = y_front.gather(1, front_pred).sum(dim=1).long()
            front_hit_sum += front_hit.sum().item()
            back_correct += (back_pred == y_back).sum().item()

            for hit in front_hit.tolist():
                hit_dist[hit] += 1

            # 留几条样本，最后打印"预测 vs 实际"
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

    random_front_hit = FRONT_COUNT * FRONT_COUNT / FRONT_NUMBER   # 6 × 6/33 ≈ 1.09
    random_back_rate = 1 / BACK_NUMBER                            # 1/16 = 6.25%

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
    print("        本项目的重点是跑通「数据 → 模型 → 训练 → 评估」这条链路。")


# todo 6 主程序
def main():
    train_dataset, val_dataset, test_dataset, feature_number = create_dataset()

    loss_epoch_train, loss_epoch_valid = train(train_dataset, val_dataset, feature_number)

    plot_loss(loss_epoch_train, loss_epoch_valid)

    test_fn(test_dataset, feature_number)


if __name__ == '__main__':
    main()
