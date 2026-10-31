"""Fetch China Welfare Lottery Double Color Ball (SSQ) draw history.

Public functions for other modules:
    get_latest_draw()              Fetch the latest draw; do not write a CSV.
    get_recent_draws(count=10)      Fetch recent draws; do not write a CSV.
    update_history_csv(csv_path)    Fetch all history and update the local CSV.

The website returns draws newest-first. CSV files are also saved newest-first.
"""

import argparse
import csv
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as http_requests
    USE_CURL_CFFI = True
except ImportError:
    # Standard requests is a fallback for environments without curl_cffi.
    import requests as http_requests
    USE_CURL_CFFI = False


BASE_URL = "https://datachart.500.com"
HISTORY_PAGE_URL = f"{BASE_URL}/ssq/history/history.shtml"
HISTORY_DATA_URL = f"{BASE_URL}/ssq/history/newinc/history.php"
DEFAULT_FILENAME = "fc_lottery_data.csv"
FRONT_COUNT = 6
BACK_COUNT = 1
FRONT_MAX = 33
BACK_MAX = 16
CSV_HEADERS = ["\u671f\u53f7", "\u524d\u533a", "\u540e\u533a"]


def get_page(url):
    """Request a page and return a BeautifulSoup object."""
    request_options = {"timeout": 30}
    if USE_CURL_CFFI:
        request_options["impersonate"] = "chrome"
    else:
        request_options["headers"] = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )
        }
    response = http_requests.get(url, **request_options)
    response.raise_for_status()
    response.encoding = "gb2312"
    return BeautifulSoup(response.text, "lxml")


def _parse_draw_rows(soup):
    """Parse valid draw rows from a history page."""
    tbody = soup.find("tbody", id="tdata")
    if tbody is not None:
        rows = tbody.find_all("tr")
    else:
        chart = soup.find("div", class_="chart")
        rows = chart.find_all("tr") if chart is not None else soup.find_all("tr")

    expected_count = 1 + FRONT_COUNT + BACK_COUNT
    records = []
    seen_issues = set()

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < expected_count:
            continue

        values = [re.sub(r"\s+", "", cell.get_text()) for cell in cells]
        issue = values[0]
        front_raw = values[1 : 1 + FRONT_COUNT]
        back_raw = values[1 + FRONT_COUNT : expected_count]

        if not issue.isdigit():
            continue
        if not all(number.isdigit() for number in front_raw + back_raw):
            continue

        front_nums = [int(number) for number in front_raw]
        back_nums = [int(number) for number in back_raw]

        if len(set(front_nums)) != FRONT_COUNT:
            continue
        if any(number < 1 or number > FRONT_MAX for number in front_nums):
            continue
        if len(back_nums) != BACK_COUNT:
            continue
        if any(number < 1 or number > BACK_MAX for number in back_nums):
            continue
        if issue in seen_issues:
            continue

        seen_issues.add(issue)
        front_nums.sort()
        records.append(
            {
                "issue": issue,
                "front": " ".join(f"{number:02d}" for number in front_nums),
                "back": " ".join(f"{number:02d}" for number in back_nums),
            }
        )

    records.sort(key=lambda item: int(item["issue"]), reverse=True)
    return records


def get_current_number():
    """Get the latest issue number published on the history page."""
    soup = get_page(HISTORY_PAGE_URL)
    end_input = soup.find("input", id="end")

    if end_input is not None and end_input.get("value", "").strip().isdigit():
        return end_input["value"].strip()

    # Fallback in case the website changes and the page itself contains draw rows.
    page_records = _parse_draw_rows(soup)
    if page_records:
        return page_records[0]["issue"]

    raise RuntimeError(
        "Unable to find the latest issue number. The data website may have changed."
    )


def fetch_history(start_issue=1, end_issue=None):
    """Fetch draws in the inclusive issue range and return newest-first records."""
    if end_issue is None:
        end_issue = get_current_number()

    start_issue = str(start_issue).strip()
    end_issue = str(end_issue).strip()
    if not start_issue.isdigit() or not end_issue.isdigit():
        raise ValueError("start_issue and end_issue must be numeric issue numbers.")
    if int(start_issue) > int(end_issue):
        raise ValueError("start_issue cannot be later than end_issue.")

    url = f"{HISTORY_DATA_URL}?start={start_issue}&end={end_issue}"
    soup = get_page(url)
    records = _parse_draw_rows(soup)

    if not records:
        raise RuntimeError(
            "No valid draw records were found. Check the website structure or network."
        )

    # Keep only records inside the requested range in case the site returns extra rows.
    low, high = int(start_issue), int(end_issue)
    records = [item for item in records if low <= int(item["issue"]) <= high]
    records.sort(key=lambda item: int(item["issue"]), reverse=True)

    if not records:
        raise RuntimeError(
            "The website response contained no records in the requested issue range."
        )
    return records


def get_latest_draw():
    """Fetch the latest draw for frontend display; this function never writes CSV."""
    latest_issue = get_current_number()

    try:
        records = fetch_history(start_issue=latest_issue, end_issue=latest_issue)
        return records[0]
    except RuntimeError:
        # Fallback for sites that do not accept a single-issue range.
        records = fetch_history(start_issue=1, end_issue=latest_issue)
        return records[0]


def get_recent_draws(count=10):
    """Fetch the most recent ``count`` draws; this function never writes CSV.

    The returned list is newest-first. The prediction/preprocessing layer should
    reverse it before building a time-series window if it expects oldest-first data.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer.")

    latest_issue = get_current_number()
    latest_num = int(latest_issue)
    year_code = latest_num // 1000
    issue_in_year = latest_num % 1000

    # Usually the requested range lies in the current year. Near a year boundary,
    # start from the previous year's first issue, then keep the latest N records.
    if issue_in_year >= count:
        start_issue = latest_num - count + 1
    else:
        start_issue = max(1, (year_code - 1) * 1000 + 1)

    try:
        records = fetch_history(start_issue=start_issue, end_issue=latest_issue)
    except RuntimeError:
        records = fetch_history(start_issue=1, end_issue=latest_issue)

    if len(records) < count and start_issue != 1:
        records = fetch_history(start_issue=1, end_issue=latest_issue)

    return records[:count]


def _resolve_csv_path(csv_path=None):
    """Resolve an explicit CSV path or default to this script's directory."""
    if csv_path is None:
        return Path(__file__).resolve().parent / DEFAULT_FILENAME

    path = Path(csv_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def update_history_csv(csv_path=None):
    """Fetch all historical draws and atomically replace the local CSV.

    Args:
        csv_path: Optional path from config.py (for example, config.DATA_PATH).
                   If omitted, save beside this script as fc_lottery_data.csv.

    Returns:
        A summary dictionary containing csv_path, count, and latest_draw.
    """
    latest_issue = get_current_number()
    records = fetch_history(start_issue=1, end_issue=latest_issue)
    output_path = _resolve_csv_path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file first, so an interrupted update does not corrupt
    # the existing CSV.
    temp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)
            for item in records:
                writer.writerow([item["issue"], item["front"], item["back"]])
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return {
        "csv_path": str(output_path),
        "count": len(records),
        "latest_draw": records[0],
    }


def _get_window_size_from_config():
    """Read WINDOW_SIZE from config.Config, with a direct .env fallback.

    The project-level config.py remains the primary configuration interface.
    The fallback is only used if Config does not expose WINDOW_SIZE/window_size.
    """
    import sys

    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    raw_value = None
    config_error = None

    # Primary path: use the same Config class as Flask and Streamlit.
    try:
        from config import Config

        config = Config()
        for attribute in ("window_size", "WINDOW_SIZE"):
            value = getattr(config, attribute, None)
            if value is not None:
                raw_value = value
                break

        # Also support projects that expose WINDOW_SIZE as a module-level value.
        if raw_value is None:
            import config as config_module

            for attribute in ("WINDOW_SIZE", "window_size"):
                value = getattr(config_module, attribute, None)
                if value is not None:
                    raw_value = value
                    break
    except Exception as exc:
        config_error = exc

    # Fallback: read WINDOW_SIZE directly from the project's .env file.
    # This makes the menu usable even if config.py has not exposed this setting yet.
    if raw_value is None:
        env_path = project_root / ".env"
        if env_path.is_file():
            with env_path.open("r", encoding="utf-8-sig") as env_file:
                for line in env_file:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    if key.strip() == "WINDOW_SIZE":
                        raw_value = value.split("#", 1)[0].strip().strip('"').strip("'")
                        break

    if raw_value is None:
        detail = f" config.py 错误：{config_error}" if config_error else ""
        raise RuntimeError(
            "无法读取 WINDOW_SIZE。请确认项目根目录的 .env 中配置了 "
            "WINDOW_SIZE=10，并确认 config.py 中的 Config 类可以读取该配置。"
            + detail
        )

    try:
        window_size = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"WINDOW_SIZE 必须是正整数，当前读取到：{raw_value!r}"
        ) from exc

    if window_size < 1:
        raise RuntimeError(
            f"WINDOW_SIZE 必须大于等于 1，当前读取到：{window_size}"
        )

    return window_size


def main():
    """Run an interactive menu for manually testing data and prediction."""
    import importlib
    import sys
    from pprint import pprint

    print("\n" + "=" * 54)
    print("福彩双色球数据工具")
    print("=" * 54)
    print("1. 获取最新中奖号码")
    print("2. 更新全部历史数据并保存到 CSV")
    print("3. 预测（自动读取 .env 中的 WINDOW_SIZE）")
    print("4. 退出程序")
    print("-" * 54)

    while True:
        try:
            choice = input("请选择操作 [1/2/3/4]: ").strip()

            if choice in {"4", "q", "Q"}:
                print("已退出。")
                break

            if choice == "1":
                print("\n正在获取最新一期开奖数据...")
                draw = get_latest_draw()
                print("\n获取成功：")
                print(f"期号：{draw['issue']}")
                print(f"前区：{draw['front']}")
                print(f"后区：{draw['back']}")
                print("\n注意：本操作只用于前端展示，不会修改 CSV。\n")

            elif choice == "2":
                print("\n正在获取全部历史开奖数据，请稍候...")
                result = update_history_csv()
                print("\n更新成功！")
                print(f"CSV 路径：{result['csv_path']}")
                print(f"保存记录数：{result['count']}")
                latest = result["latest_draw"]
                print(
                    f"最新一期：{latest['issue']}  "
                    f"前区 {latest['front']}  后区 {latest['back']}"
                )
                print()

            elif choice == "3":
                # Read the setting at prediction time rather than hardcoding 10.
                # After changing .env, restart this program to reload project config.
                window_size = _get_window_size_from_config()
                print(f"\n已读取配置：WINDOW_SIZE={window_size}")
                print(f"正在获取最近 {window_size} 期开奖数据...")
                recent_draws = get_recent_draws(window_size)
                print(
                    f"\n已获取最近 {len(recent_draws)} 期数据 "
                    "（最新期在前，供预测函数使用）："
                )
                for draw in recent_draws:
                    print(
                        f"期号 {draw['issue']}  "
                        f"前区 {draw['front']}  后区 {draw['back']}"
                    )

                project_root = str(Path(__file__).resolve().parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                try:
                    prediction_module = importlib.import_module("_04_service.predict")
                except Exception as exc:
                    print(
                        "\n已获取最近开奖数据，但暂时无法导入预测模块 "
                        "_04_service/predict.py。"
                    )
                    print("请检查该文件及其依赖是否已准备好。")
                    print(f"详细信息：{exc}\n")
                    continue

                predict_function = getattr(
                    prediction_module, "predict_all_models", None
                )
                if not callable(predict_function):
                    print(
                        "\n预测模块已导入，但没有找到 "
                        "predict_all_models(recent_draws) 函数。"
                    )
                    print(
                        "请在 _04_service/predict.py 中提供该接口，"
                        "或对齐现有函数名称。\n"
                    )
                    continue

                try:
                    print("\n正在运行预测模型...")
                    prediction_result = predict_function(recent_draws)
                    print("\n预测函数返回结果：")
                    pprint(prediction_result, sort_dicts=False)
                    print()
                except Exception as exc:
                    print("\n预测函数运行失败。")
                    print(f"详细信息：{exc}\n")
                    print(
                        "提示：如果你把 WINDOW_SIZE 从 10 改成 20，"
                        "需要使用窗口长度为 20 的数据重新训练模型；"
                        "仅增加输入期数，不能让按 10 期训练的模型自动适配。\n"
                    )

            else:
                print("无效选项，请输入 1、2、3 或 4。\n")

        except (RuntimeError, ValueError) as exc:
            print(f"操作失败：{exc}\n")
        except KeyboardInterrupt:
            print("\n用户中断，已退出。")
            break
        except Exception as exc:
            print(f"发生未预期的错误：{exc}\n")


if __name__ == "__main__":
    main()
