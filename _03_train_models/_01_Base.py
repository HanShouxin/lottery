"""
 随机预测（Base 基线）

 这个文件只有几十行，但它是整个项目的"下限"：
     完全不训练、不看历史数据，直接从 33 个红球里随机抽 6 个、从 16 个蓝球里随机抽 1 个。

 为什么一定要有一条基线：
     5 个神经网络训练完之后，怎么判断它们"有用"？不能只看 loss 降了多少，
     必须和一个"什么都不学"的对手比。实测结果是：
     5 个模型的前区命中在 1.05~1.11 之间，而随机基线是 1.09 —— 全都在同一个水平线上。
     这个对比本身就是本项目最重要的一课：loss 低 ≠ 有预测能力。

 说明：本文件没有 2244 骨架，因为它不涉及训练 —— 基线就是要"什么都不会"。
"""

import random
import sys
from pathlib import Path

# 让脚本在任何目录下运行都能导入项目根目录的 config
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import config


# 福彩双色球随机预测
def fc_base(seed=None):
    """
    随机摇一注，返回 {"front": [6 个号码], "back": 1 个号码}

    参数 seed:
        不传就不设种子 —— 每次都不同，这才是"随机基线"该有的样子；
        传了就把种子固定下来，方便复现和调试。
    """
    if seed is not None:
        random.seed(seed)

    # random.sample：从 1~33 里"不重复"地抽 6 个（这一步就保证了不会出现重号）
    ssq_red = sorted(random.sample(range(1, config.front_number + 1), config.front_count))

    # 蓝球从 1~16 里抽 1 个
    ssq_blue = random.randint(1, config.back_number)

    # sorted 之后返回，方便查看
    return {"front": ssq_red, "back": ssq_blue}


if __name__ == '__main__':
    # 福彩随机预测
    result = fc_base()

    print("福彩双色球红球号码：", result["front"])
    print("福彩双色球蓝球号码：", result["back"])
    print("\n随机基线的前区命中期望 = 6 × 6/33 ≈ 1.09 个，后区命中率 = 1/16 = 6.25%")
