'''
实现一个 streamlit 框架的前端页面，调用 flask 提供的 web 服务
（前后端分离：浏览器 ←→ streamlit ←→ flask ←→ 模型 / 爬虫）

主要步骤
    0.创建配置对象、拼出接口地址
    1.配置 streamlit 页面
    2.向 flask 后端发送请求
        2.1  GET  /latest   获取最新中奖号码
        2.2  POST /predict  预测下一期，并按模型结果统计重复号、组合成复式买号

启动顺序（两个终端，缺一不可）
    终端 1：python _05_web/app_flask.py
    终端 2：streamlit run _05_web/app_streamlit.py

为什么文件名不叫 streamlit.py：
    `streamlit run streamlit.py` 时脚本目录会进 sys.path 最前面，
    `import streamlit` 会导入脚本自己，导致 ImportError。所以加了 app_ 前缀。
'''

import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import requests
import streamlit as st

# todo 0 让脚本在任何目录下运行都能导入项目根目录的模块
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import config

# 拼出接口地址：http://127.0.0.1:5000
API_BASE = f"http://{config.flask_host}:{config.flask_port}"


# todo 1 配置页面
st.set_page_config(
    page_title="福彩双色球预测",
    page_icon="🎯",
    layout="centered",       # 居中窄版，号码排成一行更整齐
)

# todo 1.1 样式
# 卡片 + 号码球都用纯 HTML 画，样式集中在这里，改颜色只改这一处
#
# 注意：这里必须用 st.html()，不能用 st.markdown(..., unsafe_allow_html=True)。
#   原因：浏览器解析 HTML 时会把 <style> 节点"挪"到别处，
#   而 st.markdown 的内容是交给 React 管的，重跑时 React 按自己记的子节点列表去删，
#   发现对不上就抛 NotFoundError: 无法对 "Node" 执行 "removeChild"。
#   st.html 有自己的容器、不参与 React 的 diff，所以没有这个问题。
st.html("""
<style>
.ssq-hero { padding: 6px 0 18px; }
.ssq-hero h1 {
    font-size: 26px; font-weight: 600; color: #1f1f1f;
    margin: 0 0 8px 0; letter-spacing: .5px;
}
.ssq-hero p { color: #8f8f8f; font-size: 13px; margin: 0; line-height: 1.7; }

.ssq-card {
    background: #ffffff; border: 1px solid #ececec; border-radius: 12px;
    padding: 14px 18px 10px; margin: 14px 0 2px;
}
.ssq-card-title {
    font-size: 13px; color: #8f8f8f; margin-bottom: 12px;
    display: flex; justify-content: space-between;
}
.ssq-card-title span:last-child { color: #c2c2c2; font-size: 12px; }

.ssq-row {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 0; border-bottom: 1px solid #f6f6f6; white-space: nowrap;
}
.ssq-row:last-child { border-bottom: none; }
.ssq-head { color: #a8a8a8; font-size: 12px; padding-bottom: 6px; }
.ssq-head span { font-weight: 400; }

.ssq-idx { width: 22px; font-size: 12px; color: #b8b8b8; font-family: Consolas, monospace; }
.ssq-issue { width: 50px; font-size: 13px; color: #444; font-family: Consolas, monospace; }
.ssq-model { width: 50px; font-size: 13px; color: #2f2f2f; font-weight: 500; }
.ssq-sub { width: 96px; font-size: 12px; color: #a8a8a8; }

.ssq-balls { display: flex; gap: 6px; width: 200px; }
.ssq-ball {
    display: inline-block; width: 28px; height: 28px; line-height: 28px;
    text-align: center; border-radius: 50%; color: #fff;
    font-family: Consolas, monospace; font-size: 12px; flex: none;
}
.ssq-red  { background: #e4393c; }
.ssq-blue { background: #1a73e8; }
.ssq-plus { width: 14px; text-align: center; color: #cfcfcf; font-size: 12px; }
.ssq-tag  { font-size: 11px; color: #c2c2c2; margin-left: 4px; }

.ssq-big { font-size: 15px; color: #2f2f2f; margin-bottom: 12px; font-family: Consolas, monospace; }
.ssq-empty { color: #b0b0b0; font-size: 13px; }

.ssq-sec { font-size: 12px; color: #a8a8a8; margin: 14px 0 8px; }
.ssq-chips { display: flex; flex-wrap: wrap; gap: 10px; }
.ssq-chip { display: inline-flex; align-items: center; gap: 5px; margin-right: 6px; }
.ssq-count { font-size: 12px; color: #a0a0a0; font-family: Consolas, monospace; }
.ssq-combo { display: flex; align-items: center; gap: 6px; padding: 3px 0; }
.ssq-rank { width: 24px; font-size: 12px; color: #c8c8c8; font-family: Consolas, monospace; }
.ssq-note { font-size: 12px; color: #9a9a9a; line-height: 1.9; }
.ssq-all { column-count: 3; column-gap: 20px;
    font-family: Consolas, monospace; font-size: 12px; color: #555; line-height: 2; }
.ssq-divider { height: 1px; background: #f2f2f2; margin: 16px 0 12px; }

.ssq-foot {
    color: #b0b0b0; font-size: 12px; line-height: 1.8;
    margin-top: 22px; padding-top: 16px; border-top: 1px solid #f0f0f0;
}
</style>
""")


# todo 2 向后端发请求
# 爬虫每次都要访问 500.com，耗时且容易被限流，所以给结果加缓存：
# ttl=60 表示 60 秒内重复点击直接复用上次结果，不会重复抓取

# 本机请求必须绕过系统代理：
#   国内电脑上 Clash / V2Ray 之类常把 http_proxy 设成 127.0.0.1:xxxx，
#   这时 requests 访问 127.0.0.1:5000 也会被塞进代理，结果是 502 Bad Gateway，
#   而且报错信息完全不提示"代理"两个字，非常难查。显式置 None 即可跳过。
LOCAL_NO_PROXY = {"http": None, "https": None}


@st.cache_data(ttl=config.cache_ttl, show_spinner=False)
def api_latest():
    resp = requests.get(f"{API_BASE}/latest", timeout=15, proxies=LOCAL_NO_PROXY)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=config.cache_ttl, show_spinner=False)
def api_predict(window_size):
    resp = requests.post(f"{API_BASE}/predict", json={"window_size": window_size},
                         timeout=120, proxies=LOCAL_NO_PROXY)
    resp.raise_for_status()
    return resp.json()


# todo 3 渲染小工具
def ball(number, color="red"):
    """单个号码球，color: red / blue"""
    return f'<span class="ssq-ball ssq-{color}">{number:02d}</span>'


def red_balls(numbers):
    """6 个前区红球，升序"""
    return "".join(ball(n) for n in sorted(numbers))


def full_numbers(front, back):
    """一组完整号码：6 红 + 1 蓝"""
    return (f'<span class="ssq-balls">{red_balls(front)}</span>'
            f'<span class="ssq-plus">+</span>{ball(back, "blue")}')


def latest_card(draw, elapsed):
    """最新一期：期号 + 一组号码"""
    front = [int(n) for n in draw["front"].split()]
    back = int(draw["back"])

    return (
        '<div class="ssq-card notranslate">'
        '<div class="ssq-card-title"><span>最新一期开奖</span>'
        f'<span>耗时 {elapsed:.0f} ms</span></div>'
        f'<div class="ssq-big">第 {draw["issue"]} 期</div>'
        f'<div class="ssq-row" style="border:none;padding:0;">{full_numbers(front, back)}</div>'
        '</div>'
    )


def compute_consensus(predictions, pool_limit=10):
    """
    把 6 个模型的结果做聚合，生成"共识号码"和"复合买号"的数据部分。

        1 统计每个号码被几个模型选中 —— 被多个模型同时选中的就是"重复号"
        2 用重复号组成复式池，穷举出所有「6 个前区号」的组合
        3 每个组合再配后区候选，就是一张可下注的复式票

    注意：重复只代表模型倾向一致，不代表更可能中奖。
          6 个模型的期望命中都等于随机水平，这里只是演示"结果聚合"这件事。

    返回 None 表示没有任何可用模型。
    """
    valid = [item for item in predictions.values() if item.get("ok")]
    if not valid:
        return None

    # 统计每个号码被选中的次数
    front_count = Counter()
    back_count = Counter()
    for item in valid:
        front_count.update(item["front"])
        back_count[item["back"]] += 1

    # 重复号：出现 >= 2 次，按次数降序、号码升序
    front_rep = sorted(((n, c) for n, c in front_count.items() if c >= 2),
                       key=lambda x: (-x[1], x[0]))
    back_rep = sorted(((n, c) for n, c in back_count.items() if c >= 2),
                      key=lambda x: (-x[1], x[0]))

    # 复式池：优先用重复号；超过 pool_limit 个就按次数截断，避免组合爆炸
    pool = [n for n, _ in front_rep][:pool_limit]

    filled = False
    if len(pool) < 6:
        # 重复号不足 6 个，用整体高频号补齐，否则凑不出一注
        for n, _ in sorted(front_count.items(), key=lambda x: (-x[1], x[0])):
            if n not in pool:
                pool.append(n)
            if len(pool) >= 6:
                break
        filled = True

    # 后区：优先重复号；没有重复号就取出现最多的那个
    blues = [n for n, _ in back_rep] or [back_count.most_common(1)[0][0]]

    # 穷举 C(池子大小, 6)，按"共识强度"（6 个号码次数之和）降序排列
    combos = sorted(itertools.combinations(sorted(pool), 6),
                    key=lambda combo: (-sum(front_count[n] for n in combo), combo))

    return {
        "model_count": len(valid),
        "front_rep": front_rep,
        "back_rep": back_rep,
        "pool": pool,
        "blues": blues,
        "combos": combos,
        "total": len(combos) * len(blues),
        "filled": filled,
    }


def consensus_card(data, show_top=10):
    """共识号码 + 复式说明 + 前若干注组合"""
    def chips(items, color):
        if not items:
            return '<span class="ssq-note">各模型无重复</span>'
        return "".join(
            f'<span class="ssq-chip">{ball(n, color)}'
            f'<span class="ssq-count">×{c}</span></span>'
            for n, c in items
        )

    # 后区只有 1 个候选时，把蓝球直接拼到每一注后面，这样每行就是完整的一张票
    one_blue = (f'<span class="ssq-plus">+</span>{ball(data["blues"][0], "blue")}'
                if len(data["blues"]) == 1 else "")

    combo_rows = "".join(
        '<div class="ssq-combo">'
        f'<span class="ssq-rank">{i:02d}</span>'
        f'<span class="ssq-balls">{red_balls(combo)}</span>'
        f'{one_blue}'
        '</div>'
        for i, combo in enumerate(data["combos"][:show_top], 1)
    )

    blue_chips = "".join(ball(n, "blue") for n in data["blues"])

    parts = [
        '<div class="ssq-card notranslate">',
        '<div class="ssq-card-title"><span>共识号码与复合买号</span>',
        f'<span>统计 {data["model_count"]} 个模型</span></div>',

        '<div class="ssq-sec">前区重复号码</div>',
        f'<div class="ssq-chips">{chips(data["front_rep"], "red")}</div>',

        '<div class="ssq-sec">后区重复号码</div>',
        f'<div class="ssq-chips">{chips(data["back_rep"], "blue")}</div>',

        '<div class="ssq-divider"></div>',

        (f'<div class="ssq-note">复式：前区 {len(data["pool"])} 选 6 ＝ '
         f'{len(data["combos"])} 注，每注后区配 {data["blues"][0]:02d}'
         f'，共 <b>{data["total"]} 注</b></div>'
         if len(data["blues"]) == 1 else
         f'<div class="ssq-note">复式：前区 {len(data["pool"])} 选 6 ＝ '
         f'{len(data["combos"])} 注　×　后区 {len(data["blues"])} 个'
         f'（每注取其中 1 个）＝ <b>{data["total"]} 注</b></div>'),
        f'<div class="ssq-note">后区候选：{blue_chips}</div>',
    ]

    if data["filled"]:
        parts.append('<div class="ssq-note">（重复号不足 6 个，已按出现次数补齐）</div>')

    parts += [
        f'<div class="ssq-sec">按共识强度排序，前 '
        f'{min(show_top, len(data["combos"]))} 注</div>',
        combo_rows,
        '</div>',
    ]

    return "".join(parts)


def combo_list_html(data):
    """全部组合，用三栏纯文本排布，塞进 expander 里看"""
    lines = "".join(
        f'<div>{" ".join(f"{n:02d}" for n in combo)}</div>'
        for combo in data["combos"]
    )
    return f'<div class="ssq-all notranslate">{lines}</div>'


# 模型名 → (简称, 中文说明)
MODEL_LABELS = [
    ("base", "Base", "随机基线"),
    ("ann", "ANN", "全连接"),
    ("cnn", "CNN", "二维卷积"),
    ("rnn", "RNN", "循环网络"),
    ("lstm", "LSTM", "长短期记忆"),
    ("gru", "GRU", "门控循环"),
]


def predict_card(predictions):
    """六个模型的预测号码，每行一个模型"""
    head = ('<div class="ssq-row ssq-head">'
            '<span class="ssq-model">模型</span>'
            '<span class="ssq-sub"></span>'
            '<span class="ssq-balls">预测号码（6 红 + 1 蓝）</span>'
            '</div>')

    rows = []
    for key, short, desc in MODEL_LABELS:
        item = predictions.get(key, {"ok": False, "error": "无结果"})

        if item.get("ok"):
            cell = full_numbers(item["front"], item["back"])
        else:
            cell = f'<span class="ssq-empty">{item.get("error", "不可用")}</span>'

        rows.append(
            '<div class="ssq-row">'
            f'<span class="ssq-model">{short}</span>'
            f'<span class="ssq-sub">{desc}</span>'
            f'{cell}'
            '</div>'
        )

    return ('<div class="ssq-card notranslate">'
            '<div class="ssq-card-title"><span>下一期预测</span>'
            '<span>6 个模型独立给出</span></div>'
            + head + "".join(rows) +
            '</div>')


# todo 4 页面主体
st.html(
    '<div class="ssq-hero notranslate">'
    '<h1>福彩双色球号码预测</h1>'
    '<p>实时抓取最近开奖数据，分别送入 ANN / CNN / RNN / LSTM / GRU 五个模型，'
    '与随机基线对照。</p>'
    '</div>'
)

# todo 4.1 两个主按钮
col_left, col_right = st.columns(2)

with col_left:
    click_latest = st.button("获取最新中奖号码", use_container_width=True)

with col_right:
    click_predict = st.button("预测下一期", type="primary", use_container_width=True)


# todo 4.2 接口 1：最新一期
if click_latest:
    with st.spinner("正在抓取最新一期…"):
        try:
            start = time.time()
            payload = api_latest()
            elapsed = (time.time() - start) * 1000

            if payload.get("ok"):
                st.session_state["latest"] = (payload["data"], elapsed)
            else:
                st.session_state.pop("latest", None)
                st.error(payload.get("error", "获取失败"))

        except requests.exceptions.RequestException as exc:
            st.session_state.pop("latest", None)
            st.error("连不上后端服务，请先在另一个终端运行：python _05_web/app_flask.py")
            st.caption(f"详细信息：{exc}")


# todo 4.3 接口 2：预测下一期
if click_predict:
    with st.spinner("正在抓取数据并推理…"):
        try:
            start = time.time()
            payload = api_predict(config.window_size)
            elapsed = (time.time() - start) * 1000

            if payload.get("ok"):
                st.session_state["prediction"] = (payload["data"], elapsed)
            else:
                st.session_state.pop("prediction", None)
                st.error(payload.get("error", "预测失败"))

        except requests.exceptions.RequestException as exc:
            st.session_state.pop("prediction", None)
            st.error("连不上后端服务，请先在另一个终端运行：python _05_web/app_flask.py")
            st.caption(f"详细信息：{exc}")


# todo 4.4 结果区
if "latest" in st.session_state:
    draw, elapsed = st.session_state["latest"]
    st.html(latest_card(draw, elapsed))

if "prediction" in st.session_state:
    result, elapsed = st.session_state["prediction"]

    st.html(predict_card(result["predictions"]))

    consensus = compute_consensus(result["predictions"])
    if consensus:
        st.html(consensus_card(consensus))
        if len(consensus["combos"]) > 10:
            with st.expander(f"查看全部 {len(consensus['combos'])} 注前区组合"):
                st.html(combo_list_html(consensus))

    st.caption(f"窗口 {result['window'][-1]['issue']} ~ {result['latest_issue']}　|　"
               f"推理耗时 {elapsed:.0f} ms")

if "latest" not in st.session_state and "prediction" not in st.session_state:
    st.html('<div class="ssq-card notranslate"><span class="ssq-empty">'
            '点击上方按钮开始。预测会自动抓取最近 '
            f'{config.window_size} 期开奖数据。</span></div>')


# todo 5 页脚
st.html(
    '<div class="ssq-foot notranslate">'
    '数据来源：500.com 实时抓取，不写入本地文件<br>'
    '双色球每期开奖是独立随机事件，任何模型的期望命中都等于随机水平；'
    '本页面展示的是「取数 → 编码 → 推理 → 展示」的完整工程链路。'
    '</div>'
)

# 缓存导致 60 秒内重复点击拿到的还是旧数据，这里留一个手动刷新入口
_, mid, _ = st.columns([1, 1, 1])
with mid:
    if st.button("重新抓取（清缓存）", use_container_width=True):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
