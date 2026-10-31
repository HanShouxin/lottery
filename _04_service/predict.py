"""
预测服务：加载训练好的模型，用最新 N 期开奖数据出号码

这一层是「前端」和「模型」之间唯一的桥：
    前端   → 调 predict_next()        拿 6 组号码去渲染
    本文件 → 调 lottery_history       抓最新 10 期（不落盘）
           → 调 process_data         把号码编码成模型输入
           → 调 5 个 .pth            前向推理
           → 选出前区 6 个 / 后区 1 个

对外一共 4 个函数：
    fetch_latest()                       取最新一期（只展示，不落盘）
    fetch_recent(n=None)                 取最近 n 期（默认 .env 的 WINDOW_SIZE）
    predict_all_models(recent_draws)     传入最近 n 期 → 6 个模型的预测结果
    predict_next()                       上面两步串起来，前端一键调用

为什么要做模型缓存：
    一个 .pth 加载要几十毫秒，每次请求都重新 load + 构造模型太浪费。
    Streamlit 每次点按钮都会重跑脚本，所以模型只加载一次、放在模块级字典里复用。

注意：本文件绝不自己写 one-hot 编码。
      编码全部走 process_data.make_predict_input()，和训练时是同一份实现。
      这条纪律能避免"训练和预测用了不同的编码，结果不报错但全错"。
"""

import contextlib
import io
import sys
from importlib import import_module
from pathlib import Path

import torch

# todo 0 让本文件在任何目录下运行都能导入项目根目录的模块
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import config
from _01_data.lottery_history import get_latest_draw, get_recent_draws
from _01_data.process_data import make_predict_input


# todo 0.1 六个模型的名字（顺序就是前端展示的顺序）
MODEL_NAMES = ["base", "ann", "cnn", "rnn", "lstm", "gru"]

# 模型名 → 训练脚本所在的模块路径
# (名字带数字开头，不能直接 import，用 import_module 按字符串导入)
MODEL_SCRIPTS = {
    "ann": "_03_train_models._02_ANN",
    "cnn": "_03_train_models._03_CNN",
    "rnn": "_03_train_models._04_RNN",
    "lstm": "_03_train_models._05_LSTM",
    "gru": "_03_train_models._06_GRU",
}

# 模型名 → _02_models/xxx_best.pth
MODEL_FILES = {name: f"{name}_best.pth" for name in MODEL_SCRIPTS}


# todo 1 模型缓存
# 第一次调用时加载，之后所有请求复用同一份
_MODULE_CACHE = {}     # 训练脚本模块（为了拿到里面的 myModel 类）
_MODEL_CACHE = {}      # 已经加载好权重的模型实例


def _device():
    """决定推理用 CPU 还是 GPU，规则和训练脚本一致"""
    if config.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _import_script(name):
    """
    导入训练脚本，拿到里面的 myModel 类。

    为什么要把输出吞掉：
        训练脚本的模块顶层会 print("使用设备：cpu")、CNN 还会打印特征图尺寸。
        这些在训练时是有用的提示，但在 Web 服务的日志里就是噪音，所以这里临时屏蔽。
        （真正的报错走 stderr，不会被吞掉）
    """
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]

    with contextlib.redirect_stdout(io.StringIO()):
        module = import_module(MODEL_SCRIPTS[name])

    _MODULE_CACHE[name] = module
    return module


def _build_model(name, module):
    """
    按各训练脚本里 __init__ 的签名构造模型。

    这是唯一需要"记住每个模型输入形状"的地方：
        ann        输入是拉平的 490 维
        cnn        需要 (B, 1, 10, 49)
        rnn/lstm/gru  直接吃 (B, 10, 49)
    """
    if name == "ann":
        return module.myModel(config.draw_dim * config.window_size)
    if name == "cnn":
        return module.myModel(config.window_size, config.draw_dim)
    return module.myModel(config.draw_dim)


def _model_input(name, seq, flat):
    """同一个窗口，按模型需要换个形状"""
    if name == "ann":
        return flat
    if name == "cnn":
        return seq.unsqueeze(1)     # (1, 10, 49) → (1, 1, 10, 49)
    return seq                       # (1, 10, 49)


def load_model(name):
    """
    加载单个模型（带缓存）。

    返回 (model, error)：
        model 为 None 表示这个模型不可用，error 里写明原因
        （比如还没训练、文件丢了）——前端可以只显示其他模型的结果，不影响整体
    """
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name], ""

    path = config.model_dir / MODEL_FILES[name]
    if not path.is_file():
        return None, f"模型文件不存在：{path.name}，请先运行 {MODEL_SCRIPTS[name]}.py"

    try:
        module = _import_script(name)
        # 构造模型时也要屏蔽：CNN 的 __init__ 里有一句打印特征图尺寸
        with contextlib.redirect_stdout(io.StringIO()):
            model = _build_model(name, module)
        # 先 load 再 eval：忘了 load 的话模型是随机初始化的，
        # 照样能输出 6 个号码，但结果毫无意义，而且不会报错
        model.load_state_dict(torch.load(path, map_location=_device()))
        model.to(_device())
        model.eval()
    except Exception as exc:
        return None, f"加载失败：{exc}"

    _MODEL_CACHE[name] = model
    return model, ""


# todo 2 取数（爬虫的薄封装，方便前端只依赖这一个文件）
def fetch_latest():
    """最新一期开奖，只用于前台展示，不写 CSV"""
    return get_latest_draw()


def fetch_recent(n=None):
    """最近 n 期，返回最新在前；n 默认取 .env 的 WINDOW_SIZE"""
    return get_recent_draws(n or config.window_size)


# todo 3 核心：把最近 n 期喂给 6 个模型
def predict_all_models(recent_draws):
    """
    参数 recent_draws：
        get_recent_draws() 的返回，最新一期在最前面，长度必须等于 WINDOW_SIZE

    返回：
        {
          "window": [ {"issue","front","back"}, ... ],      # 原样回传，前端用来显示"喂给模型的 10 期"
          "predictions": {
              "base": {"front": [4,7,...], "back": 9, "ok": True, "error": ""},
              "ann":  {...},
              ...
          },
          "window_size": 10,
          "latest_issue": "26124",
        }
    """
    window_size = config.window_size

    if len(recent_draws) != window_size:
        raise ValueError(f"需要最近 {window_size} 期，实际传入 {len(recent_draws)} 期")

    # 编码只做一次，6 个模型共用
    # 注意 make_predict_input 内部会 reversed()：爬虫给的是"最新在前"，模型要"最旧在前"
    seq = torch.from_numpy(make_predict_input(recent_draws)).to(_device())            # (1, 10, 49)
    flat = torch.from_numpy(make_predict_input(recent_draws, flatten=True)).to(_device())  # (1, 490)

    predictions = {}

    for name in MODEL_NAMES:

        # todo 3.1 Base：不训练，纯随机
        if name == "base":
            from _03_train_models._01_Base import fc_base
            result = fc_base()
            predictions[name] = {"front": result["front"], "back": result["back"],
                                 "ok": True, "error": ""}
            continue

        # todo 3.2 神经网络
        model, error = load_model(name)
        if model is None:
            predictions[name] = {"front": [], "back": 0, "ok": False, "error": error}
            continue

        with torch.no_grad():
            pred_front, pred_back = model(_model_input(name, seq, flat))

            # 前区：33 个得分里最高的 6 个；topk 的下标天然不重复
            front = sorted(int(i) + 1 for i in pred_front.topk(config.front_count, dim=1).indices[0])
            # 后区：16 个得分里最大的那个
            back = int(pred_back.argmax(dim=1)[0]) + 1

        predictions[name] = {"front": front, "back": back, "ok": True, "error": ""}

    return {
        "window": list(recent_draws),        # 最新在前，前端直接按顺序渲染
        "predictions": predictions,
        "window_size": window_size,
        "latest_issue": recent_draws[0]["issue"],
    }


# todo 4 前端一键调用：取数 + 预测
def predict_next():
    """抓最近 N 期 → 6 个模型出号码。Flask 的 /predict 直接调这个"""
    draws = fetch_recent()
    return predict_all_models(draws)


# todo 5 命令行自测
def _demo():
    print("=" * 72)
    print("predict.py 自测")
    print("=" * 72)

    # todo 5.1 最新一期
    data_path = config.data_path
    print(f"\n【1】最新一期（爬虫实时抓取，不落盘）")
    latest = fetch_latest()
    print(f"  期号 {latest['issue']}   前区 {latest['front']}   后区 {latest['back']}")
    print(f"  本地 CSV 最后更新于：{data_path.name}")

    # todo 5.2 最近 N 期
    print(f"\n【2】最近 {config.window_size} 期（喂给模型的窗口）")
    draws = fetch_recent()
    for i, draw in enumerate(draws):
        mark = "← 最新" if i == 0 else ("← 最旧" if i == len(draws) - 1 else "")
        print(f"  {i + 1:>2}. {draw['issue']}  {draw['front']}  + {draw['back']}  {mark}")

    # todo 5.3 六个模型预测
    print(f"\n【3】各模型预测（下一期）")
    result = predict_all_models(draws)
    for name in MODEL_NAMES:
        item = result["predictions"][name]
        if not item["ok"]:
            print(f"  {name.upper():<5} 不可用：{item['error']}")
            continue
        front = " ".join(f"{n:02d}" for n in item["front"])
        print(f"  {name.upper():<5} {front}  +  {item['back']:02d}")

    print("\n说明：双色球每期独立随机，六组号码的期望命中都是随机水平。")
    print("      本项目展示的是「取数 → 编码 → 推理 → 展示」这条完整链路。")


if __name__ == '__main__':
    _demo()
