'''
演示
   用 Flask 把「预测服务」暴露成 HTTP 接口，Streamlit 那边通过 requests 调用

启动：
   python _05_web/app_flask.py
   默认地址 http://127.0.0.1:5000（host / port 来自 .env）

三个接口：
   GET  /latest    取最新一期开奖（只展示，不落盘）
   POST /predict   抓最近 N 期 → 6 个模型出号码
   GET  /health    健康检查，确认服务和模型都在

为什么文件名不叫 flask.py：
   如果用 flask.py，运行 `python _05_web/flask.py` 时脚本所在目录会被加到 sys.path 最前面，
   一句 `from flask import Flask` 会导入脚本自己，直接报 ImportError。
   streamlit.py 同理。所以两个文件都加了 app_ 前缀。

启动顺序（前后端分离，两个进程）：
   1 终端 1：python _05_web/app_flask.py                ← 后端
   2 终端 2：streamlit run _05_web/app_streamlit.py     ← 前端
'''

import sys
from pathlib import Path

from flask import Flask, jsonify, request

# todo 0 让脚本在任何目录下运行都能导入项目根目录的模块
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import config
from _04_service import predict


# 1.创建 Flask 实例
app = Flask(__name__)
# 让 jsonify 直接输出中文，而不是 \u4e2d\u6587 这种转义
app.json.ensure_ascii = False


# 2.创建路由 + 视图函数
# 路由就是一张"地址-功能"的映射表，告诉 Web 服务器：当用户访问某个 URL 时该执行什么代码

@app.route('/latest', methods=['GET'])
def latest():
    """取最新一期开奖号码。前端点「获取最新中奖号码」时调这个"""
    try:
        draw = predict.fetch_latest()
        return jsonify({'ok': True, 'data': draw})
    except Exception as exc:
        # 爬虫失败（网络问题 / 网站改版）也要返回规范 JSON，前端才好提示
        return jsonify({'ok': False, 'error': f'获取最新开奖失败：{exc}'}), 500


@app.route('/predict', methods=['POST'])
def predict_api():
    """
    预测下一期。

    请求体（可选）：{"window_size": 10}
    返回：6 个模型的号码 + 喂给模型的那 10 期原始数据
    """
    try:
        body = request.get_json(silent=True) or {}
        window_size = int(body.get('window_size') or config.window_size)

        # 抓最近 N 期（不落盘），再交给 6 个模型
        draws = predict.fetch_recent(window_size)
        result = predict.predict_all_models(draws)

        return jsonify({'ok': True, 'data': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'预测失败：{exc}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查：确认服务活着、模型文件都在"""
    status = {}
    for name in predict.MODEL_NAMES:
        if name == 'base':
            status[name] = '无需模型'
            continue
        path = config.model_dir / predict.MODEL_FILES[name]
        status[name] = 'OK' if path.is_file() else f'缺少 {path.name}'

    return jsonify({
        'ok': True,
        'window_size': config.window_size,
        'model_dir': str(config.model_dir),
        'models': status,
    })


@app.route('/', methods=['GET'])
def index():
    """根路径给个提示，避免打开浏览器看到 404 一脸茫然"""
    return jsonify({
        'service': '福彩双色球预测服务',
        'endpoints': ['GET /latest', 'POST /predict', 'GET /health'],
    })


# 启动 web API 服务
if __name__ == '__main__':

    print("=" * 66)
    print("福彩双色球预测服务")
    print("=" * 66)
    print(f"  接口地址：http://{config.flask_host}:{config.flask_port}")
    print(f"  窗口长度：{config.window_size} 期")
    print(f"  模型目录：{config.model_dir}")
    print("=" * 66)
    print("  接口：GET /latest    POST /predict    GET /health")
    print("  按 Ctrl+C 停止服务")
    print("=" * 66)

    '''注意，现在你的笔记本就是一个服务器，host 指定服务器监听的 IP 地址
       '127.0.0.1' 是本机地址，只允许本机访问

       use_reloader=False：
         debug=True 默认会开启热重载，父进程再起一个子进程，
         结果是 6 个模型被加载两遍（慢一倍），后台运行时也不方便停。这里关掉。'''
    app.run(
        host=config.flask_host,
        port=config.flask_port,
        debug=True,
        use_reloader=False,
    )
