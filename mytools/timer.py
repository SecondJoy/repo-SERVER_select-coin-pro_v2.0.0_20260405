#JoyAdded 马代代码
import contextlib
import json
import time
from datetime import datetime

import requests


@contextlib.contextmanager
def timer(msg=None, log_func=print):
    begin_time = time.perf_counter()
    yield
    time_elapsed = time.perf_counter() - begin_time
    log_func(f"{msg or 'timer'} | {time_elapsed:.2f} sec elapsed ")


def get_wechat_url(code: str = None) -> str:
    return f'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={code}'


def send_wechat_work_msg(content, url):
    if not url:
        print('未配置wechat_webhook_url，不发送信息')
        return
    try:
        data = {
            "msgtype": "text",
            "text": {
                "content": content + '\n' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        r = requests.post(url, data=json.dumps(data), timeout=10)
        print(f'调用企业微信接口返回： {r.text}')
        print('成功发送企业微信')
    except Exception as e:
        print(f"发送企业微信失败:{e}")
