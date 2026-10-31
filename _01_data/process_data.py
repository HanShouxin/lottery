"""
数据处理：把 fc_lottery_data.csv 加工成 ANN / CNN / RNN / LSTM / GRU 能直接吃的数据集

整条流水线一共 7 步，六个训练脚本都复用这里的结果：
    todo 1  读取历史数据       CSV（最新在上）→ 按期号升序的 DataFrame
    todo 2  解析单期号码       "01 02 04 17 19 30" + "09" → [1,2,4,17,19,30] + 9
    todo 3  单期编码           一期 → 49 维 one-hot（前区 33 + 后区 16）
    todo 4  整表编码           3519 期 → (3519, 49)
    todo 5  滑动窗口           连续 10 期 → 1 个样本（X）＋ 第 11 期当标签（y）
    todo 6  按时间划分         8:1:1 → 训练 / 验证 / 测试
    todo 7  打包成数据集对象    对外只暴露 3 个函数

对外只需要记住 3 个函数：
    create_dataset_ann()         →  X 是 (N, 490)  给全连接网络 ANN
    create_dataset_seq()         →  X 是 (N, 10, 49) 给 CNN / RNN / LSTM / GRU
    make_predict_input(draws)    →  (1, 10, 49)     给 predict.py 做预测

为什么编码只写一份：
    训练时用历史 CSV，预测时用爬虫抓的最新 10 期。两边的数据来源不同，
    但必须走"同一套编码规则"，否则模型看到的数字和训练时不是一个含义——
    这种情况不会报错，只会安静地算错。所以编码只保留 encode_one_draw() 一份实现。
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

# todo 0 让本文件在任何目录下运行都能 import 到项目根目录的 config
# 因为 _01_data 是子目录，Python 默认只会在当前脚本所在目录找模块，找不到 config.py
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import config


# 常用参数从配置里取出来，后面少写一层 config.xxx
FRONT_NUMBER = config.front_number      # 33
FRONT_COUNT = config.front_count        # 6
BACK_NUMBER = config.back_number        # 16
BACK_COUNT = config.back_count          # 1
WINDOW_SIZE = config.window_size        # 10
DRAW_DIM = config.draw_dim              # 49 = 33 + 16
CSV_COLUMNS = ["期号", "前区", "后区"]


# todo 1 读取历史数据
def read_history(path=None):
    """
    读入中奖记录，返回按期号【从旧到新】排好序的 DataFrame。

    CSV 文件里最新一期在最上面，而滑窗必须"用过去预测未来"，
    所以这里统一转成升序。顺序一旦反了，模型就是在用未来预测过去，
    loss 照样会下降，但整个实验失去意义——而且很难发现。
    """
    path = Path(path) if path is not None else config.data_path

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到数据文件：{path}\n"
            f"请先运行 _01_data/lottery_history.py 选第 2 项，把历史数据下载到本地。"
        )

    # dtype=str：全部按字符串读，"01" 才不会变成数字 1 丢掉前导零
    # encoding="utf-8-sig"：文件带 BOM 头，不加这个表头第一列会变成 "\ufeff期号"
    data = pd.read_csv(path, dtype=str, encoding="utf-8-sig")

    missing = [name for name in CSV_COLUMNS if name not in data.columns]
    if missing:
        raise ValueError(f"CSV 缺少列：{missing}，实际列名：{list(data.columns)}")

    data = data[CSV_COLUMNS].copy()
    data["期号"] = data["期号"].str.strip()

    if data.empty:
        raise ValueError(f"数据文件是空的：{path}")

    if not data["期号"].str.isdigit().all():
        bad = data.loc[~data["期号"].str.isdigit(), "期号"].tolist()[:5]
        raise ValueError(f"期号必须是纯数字，发现异常值：{bad}")

    # 期号形如 26122 = 2026 年第 122 期，转成整数排序即可（跨年也不会错，
    # 因为年份在前：25151 < 26001）
    data["_排序用"] = data["期号"].astype(np.int64)

    duplicated = data["_排序用"].duplicated()
    if duplicated.any():
        raise ValueError(f"期号重复：{data.loc[duplicated, '期号'].tolist()[:5]}")

    data = data.sort_values("_排序用").reset_index(drop=True)
    data = data.drop(columns=["_排序用"])

    if len(data) <= WINDOW_SIZE + 1:
        raise ValueError(
            f"历史数据只有 {len(data)} 期，不够构造窗口长度为 {WINDOW_SIZE} 的样本"
        )

    return data


# todo 2 解析单期号码
def parse_draw(front_text, back_text):
    """
    把 "01 02 04 17 19 30" 和 "09" 解析成 [1, 2, 4, 17, 19, 30] 和 9，并做合法性校验。

    校验不是多余的动作：爬虫抓下来的数据一旦混进越界号码（比如前区出现 34），
    np.zeros(33) 直接索引越界；混进重复号码则 one-hot 会少 1 个，样本静默出错。
    """
    # 正则提取所有数字，能同时兼容 "01 02 04"、"01,02,04"、"010204" 这些写法
    front = [int(n) for n in re.findall(r"\d+", str(front_text))]
    back = [int(n) for n in re.findall(r"\d+", str(back_text))]

    if len(front) != FRONT_COUNT:
        raise ValueError(f"前区应有 {FRONT_COUNT} 个号码，实际 {len(front)} 个：{front_text!r}")
    if len(set(front)) != FRONT_COUNT:
        raise ValueError(f"前区号码重复：{front_text!r}")
    if any(not 1 <= n <= FRONT_NUMBER for n in front):
        raise ValueError(f"前区号码越界（应为 1~{FRONT_NUMBER}）：{front_text!r}")

    if len(back) != BACK_COUNT:
        raise ValueError(f"后区应有 {BACK_COUNT} 个号码，实际 {len(back)} 个：{back_text!r}")
    if any(not 1 <= n <= BACK_NUMBER for n in back):
        raise ValueError(f"后区号码越界（应为 1~{BACK_NUMBER}）：{back_text!r}")

    # 前区升序，方便肉眼核对
    return sorted(front), back[0]


# todo 3 单期编码：一期 → 49 维 one-hot
def encode_one_draw(front_text, back_text):
    """
    一期开奖 → 49 维 float32 向量

        [ 前区 33 维 ][ 后区 16 维 ]
          号码 n  →  下标 n-1       后区在这里额外偏移 33

    例：前区 01 02 04 17 19 30，后区 09
        下标 0,1,3,16,18,29            = 1
        下标 33 + 8 = 41               = 1
        其余 42 个位置                  = 0

    为什么用 one-hot 而不是直接用号码本身：
        号码 1 和 2 之间没有"大小"关系，直接当连续数值喂给网络，
        模型会误以为 33 比 1 "大 33 倍"。one-hot 让每个号码独立成一个维度，
        彼此距离相等，符合彩票号码的真实语义。
    """
    front, back = parse_draw(front_text, back_text)

    vector = np.zeros(DRAW_DIM, dtype=np.float32)

    # numpy 支持用下标数组一次性赋值，比 for 循环快也更简洁
    vector[np.array(front) - 1] = 1.0

    # 后区号码 1~16 落在 33~48 这一段
    vector[FRONT_NUMBER + back - 1] = 1.0

    return vector


# todo 4 整表编码：(T,) 期 → (T, 49)
def encode_series(data):
    """
    逐期编码，返回形状 (T, 49) 的矩阵，每一行是一期。

    T = 3519，也就是 CSV 里的总期数。
    """
    vectors = []

    for _, row in data.iterrows():
        try:
            vectors.append(encode_one_draw(row["前区"], row["后区"]))
        except ValueError as exc:
            # 报错时带上期号，否则几千行里根本不知道是哪一期出了问题
            raise ValueError(f"第 {row['期号']} 期数据异常：{exc}") from exc

    return np.stack(vectors).astype(np.float32)


# todo 5 滑动窗口：连续 10 期 → 1 个样本
def make_windows(series, window_size=None):
    """
    把 (T, 49) 的时间序列切成若干个"监督学习样本"

        X[i]        = series[i-10 : i]   → 形状 (10, 49)，第 0 行最旧、第 9 行最近
        Y_front[i]  = series[i, 0:33]    → 第 i 期的前区 one-hot，形状 (33,)
        Y_back[i]   = series[i, 33:49]   → 第 i 期的后区类别下标 0~15，是个整数

    第 i 期的窗口"右端"是 i-1 期，所以不存在数据泄漏：
    预测第 i 期时，只用了第 i-10 ~ i-1 期的信息。

    样本数 = T - window_size = 3519 - 10 = 3509
    """
    window_size = window_size or WINDOW_SIZE

    if len(series) <= window_size:
        raise ValueError(
            f"数据只有 {len(series)} 期，不足以构造窗口长度为 {window_size} 的样本"
        )

    X, Y_front, Y_back = [], [], []

    for i in range(window_size, len(series)):
        # 输入：连续 window_size 期，保持时间顺序（不要打乱！）
        X.append(series[i - window_size:i])

        # 前区标签：直接复用已经编码好的那一行，保证输入输出用的是同一套编码
        Y_front.append(series[i, :FRONT_NUMBER])

        # 后区标签：one-hot 里有 1 的那个下标，转成 0~15 的类别编号，
        # 因为分类任务用 nn.CrossEntropyLoss() 时标签必须是"类别下标"，不能是 one-hot
        Y_back.append(int(np.argmax(series[i, FRONT_NUMBER:])))

    X = np.stack(X).astype(np.float32)              # (N, 10, 49)
    Y_front = np.stack(Y_front).astype(np.float32)  # (N, 33)
    Y_back = np.array(Y_back, dtype=np.int64)       # (N,)

    return X, Y_front, Y_back


# todo 6 按时间顺序划分 8:1:1
def split_data(X, Y_front, Y_back):
    """
    时间序列不能随机打乱后划分，否则验证集会混进"未来的数据"，
    评估结果虚高。这里严格按时间先后切三段：

        |<---- 训练 80% ---->|<-- 验证 10% -->|<-- 测试 10% -->|
        旧                                                       新

    N = 3509 时：训练 2807 / 验证 350 / 测试 352
    （训练集内部用 DataLoader(shuffle=True) 打乱批次顺序，那是另一回事）
    """
    n = len(X)

    # 注意 int() 是向下取整，两段边界都这样算，保证不会越界
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    if not 0 < train_end < val_end < n:
        raise ValueError(
            f"数据量 {n} 太少，无法按 {config.train_ratio}:{config.val_ratio} 划分"
        )

    train = (X[:train_end], Y_front[:train_end], Y_back[:train_end])
    val = (X[train_end:val_end], Y_front[train_end:val_end], Y_back[train_end:val_end])
    test = (X[val_end:], Y_front[val_end:], Y_back[val_end:])

    return train, val, test


# todo 7 内部：把三段 numpy 数组打包成 TensorDataset
def _to_dataset(parts):
    """
    (array, array, array) → TensorDataset

    ascontiguousarray：PyTorch 要求内存连续，切片出来的数组不一定是，
    不转换的话 torch.from_numpy() 会直接报错。
    """
    tensors = [torch.from_numpy(np.ascontiguousarray(array)) for array in parts]
    return TensorDataset(*tensors)


def build_samples(window_size=None):
    """
    内部公用流程：读 CSV → 编码 → 滑窗。
    create_dataset_ann() 和 create_dataset_seq() 都走这里，保证两者数据来源完全一致。
    """
    data = read_history()
    series = encode_series(data)
    X, Y_front, Y_back = make_windows(series, window_size)
    return data, series, X, Y_front, Y_back


# ============================================================================
# 下面 3 个是对外接口，训练脚本和预测脚本只调这几个
# ============================================================================

# todo 8 接口一：给 ANN（全连接网络）
def create_dataset_ann():
    """
    全连接网络只认二维输入，所以要把 (10, 49) 拉平成长度 490 的向量。

        X: (N, 490)   ← 10 期 × 49 维，首尾相接
        Y_front: (N, 33)
        Y_back:  (N,)

    返回：train_dataset, val_dataset, test_dataset, feature_number
    """
    _, _, X, Y_front, Y_back = build_samples()

    # (N, 10, 49) → (N, 490)
    # 第 0~48 位是第 1 期，第 49~97 位是第 2 期，……，期与期的边界靠位置隐式表达
    X = X.reshape(len(X), -1)

    train, val, test = split_data(X, Y_front, Y_back)

    print(f"样本总数 {len(X)}：训练 {len(train[0])} / 验证 {len(val[0])} / 测试 {len(test[0])}")

    return (
        _to_dataset(train),
        _to_dataset(val),
        _to_dataset(test),
        X.shape[1],          # 490
    )


# todo 9 接口二：给 CNN / RNN / LSTM / GRU（序列模型）
def create_dataset_seq():
    """
    序列模型保留时间维，所以不做 reshape，直接给 (10, 49)。

        X: (N, 10, 49)   ← 10 个时间步，每步 49 维

    各模型拿到之后自己调整维度，公共数据层不掺任何模型专属的变形：
        CNN  方案 A：x.unsqueeze(1)     → (B, 1, 10, 49)  当成 10×49 的单通道"图"
        RNN/LSTM/GRU：直接喂            batch_first=True，input_size=49

    返回：train_dataset, val_dataset, test_dataset, window_size, feature_number
    """
    _, _, X, Y_front, Y_back = build_samples()

    train, val, test = split_data(X, Y_front, Y_back)

    print(f"样本总数 {len(X)}：训练 {len(train[0])} / 验证 {len(val[0])} / 测试 {len(test[0])}")

    return (
        _to_dataset(train),
        _to_dataset(val),
        _to_dataset(test),
        X.shape[1],          # 10  序列长度
        X.shape[2],          # 49  每个时间步的特征数
    )


# todo 10 接口三：给 predict.py（爬虫数据 → 模型输入）
def make_predict_input(recent_draws, window_size=None, flatten=False):
    """
    把爬虫抓到的最近 N 期开奖，变成模型能吃的输入。

    参数 recent_draws：
        lottery_history.get_recent_draws() 返回的列表，格式是
        [{"issue": "26122", "front": "01 02 04 17 19 30", "back": "09"}, ...]
        注意：**最新的一期排在最前面**。

    参数 flatten：
        False（默认）→ (1, 10, 49)   给 CNN / RNN / LSTM / GRU
        True          → (1, 490)     给 ANN（和 create_dataset_ann() 的输入形状对齐）

    返回：ndarray，可以直接写 model(x) 做推理。

    这里是整个项目最容易埋 bug 的地方：
        训练时窗口是"旧 → 新"，如果这里忘了 reversed()，
        模型看到的是"新 → 旧"，输出照样是 6 个号码，但完全是错的，而且不报错。
    """
    window_size = window_size or WINDOW_SIZE

    if len(recent_draws) != window_size:
        raise ValueError(
            f"需要 {window_size} 期数据，实际传入 {len(recent_draws)} 期"
        )

    # 爬虫返回"最新在前"，模型要"最旧在前"，所以翻过来
    draws = list(reversed(recent_draws))

    vectors = [encode_one_draw(draw["front"], draw["back"]) for draw in draws]

    # np.stack → (10, 49)；[None, ...] 在最前面加一维 → (1, 10, 49)
    # 维度 1 是 batch 维：单条数据也要凑成"一批"，模型才肯接收
    inputs = np.stack(vectors).astype(np.float32)[None, ...]

    if flatten:
        # 和 create_dataset_ann() 里 X.reshape(len(X), -1) 完全一样的拉平方式
        inputs = inputs.reshape(len(inputs), -1)

    return inputs


# todo 11 反向解码：one-hot → 号码（自检和前端展示用）
def decode_onehot(vector):
    """
    49 维向量 → ([前区号码...], 后区号码)
    33 维向量 → ([前区号码...], None)

    阈值 0.5：one-hot 是 0/1；如果传进来的是模型输出的 logits，
    请先用 sigmoid 转成概率，或者直接对前区取 topk。
    """
    vector = np.asarray(vector, dtype=np.float32).ravel()

    if vector.size == DRAW_DIM:
        front = [i + 1 for i in range(FRONT_NUMBER) if vector[i] > 0.5]
        back = int(np.argmax(vector[FRONT_NUMBER:])) + 1
        return front, back

    if vector.size == FRONT_NUMBER:
        front = [i + 1 for i in range(FRONT_NUMBER) if vector[i] > 0.5]
        return front, None

    raise ValueError(f"只支持 {DRAW_DIM} 维或 {FRONT_NUMBER} 维向量，收到 {vector.size} 维")


# todo 12 自检：把数据从头到尾过一遍，肉眼确认没有串行
def _self_check():
    print("=" * 72)
    print("process_data.py 自检")
    print("=" * 72)

    # todo 12.1 原始数据
    data = read_history()
    print(f"\n【1】原始数据")
    print(f"  记录数：{len(data)} 期")
    print(f"  期号范围：{data['期号'].iloc[0]} → {data['期号'].iloc[-1]}（已按从旧到新排序）")
    print(f"  最新一期：{data['期号'].iloc[-1]}  "
          f"前区 {data['前区'].iloc[-1]}  后区 {data['后区'].iloc[-1]}")

    # todo 12.2 单期编码
    series = encode_series(data)
    print(f"\n【2】编码结果")
    print(f"  形状：{series.shape}   (期数, 前区{FRONT_NUMBER} + 后区{BACK_NUMBER})")
    print(f"  每行 1 的个数：应为 {FRONT_COUNT + BACK_COUNT}，实际 "
          f"{sorted({int(v) for v in series.sum(axis=1)})}")

    front, back = decode_onehot(series[-1])
    print(f"  反解最后一期：期号 {data['期号'].iloc[-1]}  前区 {front}  后区 {back}")

    # todo 12.3 滑窗
    X, Y_front, Y_back = make_windows(series)
    print(f"\n【3】滑动窗口（WINDOW_SIZE = {WINDOW_SIZE}）")
    print(f"  X        {X.shape}   (样本数, 窗口长度, 单期维度)")
    print(f"  Y_front  {Y_front.shape}")
    print(f"  Y_back   {Y_back.shape}   取值范围 {Y_back.min()}~{Y_back.max()}")

    # 一致性校验：窗口的最后一行 = 第 10 期；标签 = 它的下一期（第 11 期）
    assert np.array_equal(X[0][-1], series[WINDOW_SIZE - 1]), "窗口末行不是第 10 期"
    assert np.array_equal(Y_front[0], series[WINDOW_SIZE][:FRONT_NUMBER]), "标签不是第 11 期"

    # 连续性校验：X[1] 应该是 X[0] 整体左移一期（丢掉最旧的一期，补上新的标签期）
    assert np.array_equal(X[1][0], X[0][1]), "滑动窗口没有按顺序平移"
    print("  校验通过：窗口末行 = 第 10 期；标签 = 第 11 期；窗口按时间顺序平移")

    # todo 12.4 第一条样本的可读形态
    # 第 0 条样本用 0~9 号（第 1~10 期）做输入，标签是 10 号（第 11 期）
    label_index = WINDOW_SIZE
    print(f"\n【4】第 0 条样本长什么样")
    print(f"  输入 10 期：")
    for row, vector in enumerate(X[0]):
        f, b = decode_onehot(vector)
        mark = "← 最旧" if row == 0 else ("← 最新" if row == WINDOW_SIZE - 1 else "")
        print(f"    行{row}  期号 {data['期号'].iloc[row]}  "
              f"前区 {' '.join(f'{n:02d}' for n in f)}  后区 {b:02d}   {mark}")
    print(f"  标签     期号 {data['期号'].iloc[label_index]}  "
          f"前区 {' '.join(f'{n:02d}' for n in decode_onehot(Y_front[0])[0])}  "
          f"后区 {Y_back[0] + 1:02d}   ← 第 11 期")

    # todo 12.5 划分
    print(f"\n【5】数据集划分（{config.train_ratio}:{config.val_ratio}:{config.test_ratio}）")
    train, val, test = split_data(X, Y_front, Y_back)
    print(f"  训练 {len(train[0])} 条 / 验证 {len(val[0])} 条 / 测试 {len(test[0])} 条")

    # todo 12.6 三个对外接口的形状
    print(f"\n【6】对外接口")

    train_ds, val_ds, test_ds, ann_feature = create_dataset_ann()
    print(f"  create_dataset_ann()   train X {tuple(train_ds.tensors[0].shape)}  "
          f"Y_front {tuple(train_ds.tensors[1].shape)}  "
          f"Y_back {tuple(train_ds.tensors[2].shape)}  feature_number={ann_feature}")

    train_ds, val_ds, test_ds, seq_len, seq_feature = create_dataset_seq()
    print(f"  create_dataset_seq()   train X {tuple(train_ds.tensors[0].shape)}  "
          f"Y_front {tuple(train_ds.tensors[1].shape)}  "
          f"Y_back {tuple(train_ds.tensors[2].shape)}  "
          f"window_size={seq_len}  feature_number={seq_feature}")
    print(f"  CNN 方案 A 在模型里 x.unsqueeze(1) → "
          f"(B, 1, {seq_len}, {seq_feature})")

    # todo 12.7 预测入口：用最后 10 期现造一条输入
    # 这里模拟爬虫的返回格式（最新在前），验证"反序"这一步有没有做对
    recent_draws = [
        {"issue": row["期号"], "front": row["前区"], "back": row["后区"]}
        for _, row in data.tail(WINDOW_SIZE).iloc[::-1].iterrows()
    ]
    predict_input = make_predict_input(recent_draws)
    print(f"\n【7】预测入口 make_predict_input()")
    print(f"  输入：最近 {len(recent_draws)} 期（最新在前）"
          f"  期号 {recent_draws[0]['issue']} → {recent_draws[-1]['issue']}")
    print(f"  输出：{predict_input.shape}")

    # 关键校验：预测输入的最后一行，必须等于历史序列的最后一期
    assert np.array_equal(predict_input[0][-1], series[-1]), "预测输入的期序颠倒了"
    print("  校验通过：预测输入的最后一行 = 最新一期（期序没有颠倒）")
    print("\n全部检查通过。")


if __name__ == '__main__':
    _self_check()
