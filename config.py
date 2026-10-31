"""
项目统一配置：把 .env 里的参数读进来，供所有脚本共用

为什么要有这个文件：
    训练脚本、预测脚本、Web 前端都需要「数据路径、窗口长度、批次大小」这些参数。
    如果每个文件各写一遍，改一个参数要改 6 个地方，还特别容易漏。
    所以约定：所有可调参数只写在 .env 里，代码里一律 from config import config。

用法：
    from config import config

    print(config.window_size)          # 10
    print(config.data_path)            # D:\\CodeLab\\lottery\\_01_data\\fc_lottery_data.csv
    print(config.model_path("ann"))    # D:\\CodeLab\\lottery\\_02_models\\ann_best.pth

注意：
    .env 里的相对路径一律以「项目根目录」为基准，而不是当前工作目录（cwd）。
    否则从 _03_train_models 里面运行脚本时，就会找不到 _01_data 下的文件。
"""

from pathlib import Path


# todo 1 项目根目录
# __file__ 是当前文件 config.py 的绝对路径，.parent 就是 D:\CodeLab\lottery
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


# todo 2 读取 .env 文件
# .env 就是一行一个 KEY=VALUE 的纯文本，几十行代码就能解析，不必额外装 python-dotenv
def load_env(env_path=ENV_PATH):
    """把 .env 解析成字典 {'KEY': 'VALUE'}"""
    if not env_path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{env_path}")

    env = {}

    # utf-8-sig：兼容记事本 / Excel 保存时带上的 BOM 头
    with env_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()

            # 空行、注释行直接跳过
            if not line or line.startswith("#"):
                continue

            # 不是 KEY=VALUE 格式的行也跳过
            if "=" not in line:
                continue

            # 只切第一个 = ，值里如果还有 = 不受影响
            key, value = line.split("=", 1)

            # 去掉行尾注释，比如 BATCH_SIZE=32   # 批次大小
            value = value.split("#", 1)[0].strip()

            # 去掉可能存在的引号
            env[key.strip()] = value.strip().strip('"').strip("'")

    return env


# todo 3 类型转换小工具
# .env 里读出来全是字符串，用之前必须转成数字，否则 "10" < "9" 这种比较会出错
def _to_int(env, key, default=None):
    raw = env.get(key, default)
    if raw is None or str(raw).strip() == "":
        raise ValueError(f".env 缺少配置项：{key}")
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f".env 中 {key} 必须是整数，当前是：{raw!r}") from exc


def _to_float(env, key, default=None):
    raw = env.get(key, default)
    if raw is None or str(raw).strip() == "":
        raise ValueError(f".env 缺少配置项：{key}")
    try:
        return float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f".env 中 {key} 必须是数字，当前是：{raw!r}") from exc


def _to_path(env, key, default=None):
    """相对路径 -> 基于项目根目录的绝对路径"""
    raw = env.get(key, default)
    if raw is None or str(raw).strip() == "":
        raise ValueError(f".env 缺少配置项：{key}")

    path = Path(str(raw).strip())
    if path.is_absolute():
        return path.resolve()
    return (BASE_DIR / path).resolve()


def _to_str(env, key, default=""):
    raw = env.get(key, default)
    return str(raw).strip() if raw is not None else default


# todo 4 配置类
# 一次性把所有参数读好、转好类型、校验完，后面各个脚本直接取属性用
class Config:

    def __init__(self, env_path=ENV_PATH):

        env = load_env(env_path)

        # todo 4.1 路径配置
        self.base_dir = BASE_DIR
        self.data_path = _to_path(env, "DATA_PATH")
        self.model_dir = _to_path(env, "MODEL_DIR")

        # todo 4.2 彩票参数（双色球：前区 6 个号取自 1~33，后区 1 个号取自 1~16）
        self.front_number = _to_int(env, "FRONT_NUMBER")
        self.front_count = _to_int(env, "FRONT_COUNT")
        self.back_number = _to_int(env, "BACK_NUMBER")
        self.back_count = _to_int(env, "BACK_COUNT")

        # 一期开奖编码后的维度：33 + 16 = 49
        # 这个值到处都要用，直接在这里算出来，别在各处写死 49
        self.draw_dim = self.front_number + self.back_number

        # todo 4.3 数据参数
        self.window_size = _to_int(env, "WINDOW_SIZE")
        self.train_ratio = _to_float(env, "TRAIN_RATIO")
        self.val_ratio = _to_float(env, "VAL_RATIO")
        # 测试集比例不单独配，由 1 - 训练 - 验证 推出来，少一个能写错的地方
        self.test_ratio = round(1.0 - self.train_ratio - self.val_ratio, 10)

        # todo 4.4 训练参数
        self.batch_size = _to_int(env, "BATCH_SIZE")
        self.epochs = _to_int(env, "EPOCHS")
        self.learning_rate = _to_float(env, "LEARNING_RATE")
        self.hidden_size = _to_int(env, "HIDDEN_SIZE")
        self.seed = _to_int(env, "RANDOM_SEED")

        # 下面几项 .env 里暂时没写，先给默认值；以后想调再加到 .env 里即可
        # weight_decay / dropout 默认取得偏大：彩票这种"没有规律"的数据极容易过拟合，
        # 正则化重一点，验证 loss 才不会几轮就飞出去
        self.weight_decay = _to_float(env, "WEIGHT_DECAY", 1e-2)
        self.lr_step_size = _to_int(env, "LR_STEP_SIZE", 30)
        self.lr_gamma = _to_float(env, "LR_GAMMA", 0.5)
        self.dropout = _to_float(env, "DROPOUT", 0.5)
        self.patience = _to_int(env, "PATIENCE", 15)   # 早停：验证 loss 连续多少轮不降就停
        self.device = _to_str(env, "DEVICE", "auto")   # auto | cpu | cuda

        # 循环网络（RNN / LSTM / GRU）专用：层数越多越容易过拟合，2 层够用
        self.rnn_hidden_size = _to_int(env, "RNN_HIDDEN_SIZE", 64)
        self.rnn_num_layers = _to_int(env, "RNN_NUM_LAYERS", 2)

        # todo 4.5 前端参数
        self.flask_host = _to_str(env, "FLASK_HOST", "127.0.0.1")
        self.flask_port = _to_int(env, "FLASK_PORT", 5000)
        self.streamlit_port = _to_int(env, "STREAMLIT_PORT", 8501)
        self.cache_ttl = _to_int(env, "CACHE_TTL", 60)

        # todo 4.6 校验：配置写错了就当场报错，别等到训练到第 3 个 epoch 才崩
        self.validate()

    # todo 5 校验
    def validate(self):

        if self.window_size < 1:
            raise ValueError(f"WINDOW_SIZE 必须 >= 1，当前：{self.window_size}")

        if self.front_count > self.front_number:
            raise ValueError(
                f"FRONT_COUNT({self.front_count}) 不能大于 FRONT_NUMBER({self.front_number})"
            )

        if self.back_count > self.back_number:
            raise ValueError(
                f"BACK_COUNT({self.back_count}) 不能大于 BACK_NUMBER({self.back_number})"
            )

        if not (0 < self.train_ratio < 1):
            raise ValueError(f"TRAIN_RATIO 必须在 (0,1) 之间，当前：{self.train_ratio}")

        if not (0 < self.val_ratio < 1):
            raise ValueError(f"VAL_RATIO 必须在 (0,1) 之间，当前：{self.val_ratio}")

        if self.test_ratio <= 0:
            raise ValueError(
                "TRAIN_RATIO + VAL_RATIO 必须小于 1，当前："
                f"{self.train_ratio} + {self.val_ratio}"
            )

        if self.batch_size < 1:
            raise ValueError(f"BATCH_SIZE 必须 >= 1，当前：{self.batch_size}")

    # todo 6 路径辅助方法
    # 模型、loss 曲线的文件名在这里统一约定，6 个训练脚本就不用各写一遍
    def model_path(self, model_name):
        """ann -> _02_models/ann_best.pth"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self.model_dir / f"{model_name}_best.pth"

    def loss_path(self, model_name):
        """ann -> _02_models/ann_loss.png"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self.model_dir / f"{model_name}_loss.png"

    def ensure_dirs(self):
        """确保模型目录存在（首次运行时它可能还是空的）"""
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # todo 7 打印配置，方便排查问题
    def describe(self):
        return {
            "数据文件": str(self.data_path),
            "模型目录": str(self.model_dir),
            "前区": f"{self.front_count} 个 / 1~{self.front_number}",
            "后区": f"{self.back_count} 个 / 1~{self.back_number}",
            "单期维度": self.draw_dim,
            "窗口长度": self.window_size,
            "划分比例": f"{self.train_ratio}:{self.val_ratio}:{self.test_ratio}",
            "批次大小": self.batch_size,
            "训练轮数": self.epochs,
            "学习率": self.learning_rate,
            "随机种子": self.seed,
            "设备": self.device,
        }

    def __repr__(self):
        lines = [f"{key}：{value}" for key, value in self.describe().items()]
        return "Config(\n  " + "\n  ".join(lines) + "\n)"


# todo 8 模块级单例
# 别的文件只需要 from config import config，不用每次都 Config()
# 也兼容了 lottery_history.py 里 Config().window_size 的写法
config = Config()


if __name__ == '__main__':
    print(config)
