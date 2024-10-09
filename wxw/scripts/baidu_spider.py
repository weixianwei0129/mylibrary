#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import re
import time
import json
import socket
import urllib
import argparse
import urllib.parse
import urllib.error
import urllib.request

from tqdm import tqdm

socket.setdefaulttimeout(timeout=5)


class Crawler:
    # 睡眠时长
    __time_sleep = 0.1
    __amount = 0
    __start_amount = 0
    __counter = 0
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0', 'Cookie': ''}
    __per_page = 30

    # 获取图片url内容等
    # t 下载图片时间间隔
    def __init__(self, folder='./', t=0.1):
        self.time_sleep = t
        self.root = folder

    # 获取后缀名
    @staticmethod
    def get_suffix(name):
        m = re.search(r'\.[^\.]*$', name)
        if m.group(0) and len(m.group(0)) <= 5:
            return m.group(0)
        else:
            return '.png'

    @staticmethod
    def handle_baidu_cookie(original_cookie, cookies):
        """
        :param string original_cookie:
        :param list cookies:
        :return string:
        """
        if not cookies:
            return original_cookie
        result = original_cookie
        for cookie in cookies:
            result += cookie.split(';')[0] + ';'
        result.rstrip(';')
        return result

    # 保存图片
    def save_image(self, rsp_data, word):
        folder = os.path.join(self.root, word)
        os.makedirs(folder, exist_ok=True)
        # 判断名字是否重复，获取图片长度
        self.__counter = len(os.listdir(folder)) + 1
        pbar = tqdm(rsp_data['data'])
        for image_info in pbar:
            try:
                if 'replaceUrl' not in image_info or len(image_info['replaceUrl']) < 1:
                    continue
                obj_url = image_info['replaceUrl'][0]['ObjUrl']
                thumb_url = image_info['thumbURL']
                url = f'https://image.baidu.com/search/down?tn=download&ipn=dwnl&word=download&ie=utf8&fr=result&url={urllib.parse.quote(obj_url)}&thumburl={urllib.parse.quote(thumb_url)}'
                time.sleep(self.time_sleep)
                suffix = self.get_suffix(obj_url)
                # 指定UA和referrer，减少403
                opener = urllib.request.build_opener()
                opener.addheaders = [
                    (
                        'User-agent',
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
                    ),
                ]
                urllib.request.install_opener(opener)
                # 保存图片
                filepath = os.path.join(folder, f"{str(self.__counter).zfill(5)}{suffix}")
                urllib.request.urlretrieve(url, filepath)
                if os.path.getsize(filepath) < 5:
                    pbar.set_description(f"下载到了空文件，跳过!")
                    os.unlink(filepath)
                    continue
            except urllib.error.HTTPError as urllib_err:
                pbar.set_description(f"{urllib_err}")
                continue
            except Exception as err:
                time.sleep(1)
                pbar.set_description(f"{err}\n产生未知错误，放弃保存")
                continue
            else:
                pbar.set_description(f"图片+1,已有{self.__counter}张图片")
                self.__counter += 1
        pbar.close()
        return

    # 开始获取
    def get_images(self, word):
        search = urllib.parse.quote(word)
        # pn int 图片数
        pn = self.__start_amount
        while pn < self.__amount:
            url = f'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={search}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&word={search}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn={pn}&rn={self.__per_page}&gsm=1e&1594447993172='
            # 设置header防403
            try:
                time.sleep(self.time_sleep)
                req = urllib.request.Request(url=url, headers=self.headers)
                page = urllib.request.urlopen(req)
                self.headers['Cookie'] = self.handle_baidu_cookie(self.headers['Cookie'],
                                                                  page.info().get_all('Set-Cookie'))
                rsp = page.read()
                page.close()
            except UnicodeDecodeError as e:
                print(e)
                print('-----UnicodeDecodeErrorurl:', url)
            except urllib.error.URLError as e:
                print(e)
                print("-----urlErrorurl:", url)
            except socket.timeout as e:
                print(e)
                print("-----socket timout:", url)
            else:
                try:
                    # 解析json
                    rsp_data = json.loads(rsp, strict=False)
                except:
                    pass
                if 'data' not in rsp_data:
                    print("触发了反爬机制，自动重试！")
                else:
                    self.save_image(rsp_data, word)
                    # 读取下一页
                    print(f"下载第{pn}页")
                    pn += self.__per_page
        return

    def start(self, word, total_page=1, start_page=1, per_page=30):
        """
        爬虫入口
        :param word: 抓取的关键词
        :param total_page: 需要抓取数据页数 总抓取图片数量为 页数 x per_page
        :param start_page:起始页码
        :param per_page: 每页数量
        :return:
        """
        self.__per_page = per_page
        self.__start_amount = (start_page - 1) * self.__per_page
        self.__amount = total_page * self.__per_page + self.__start_amount
        self.get_images(word)
        print(f"{word} 下载任务结束")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folder", type=str, help="保存的路径",
        default='/home/cltx/code/pd/scripts/process',
    )
    parser.add_argument(
        "-ws", "--words", type=str, nargs='+', help="抓取关键词",
        default=['车内后排安全带'],
    )
    parser.add_argument("-tp", "--total_page", type=int, default=10, help="需要抓取的总页数")
    parser.add_argument("-sp", "--start_page", type=int, default=1, help="起始页数")
    parser.add_argument(
        "-pp", "--per_page", type=int, default=30, help="每页大小",
        choices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], nargs='?'
    )
    parser.add_argument("-d", "--delay", type=float, help="抓取延时（间隔）", default=0.05)
    args = parser.parse_args()
    crawler = Crawler(folder=args.folder, t=args.delay)
    for word in args.words:
        print(f"do [{word}] ....")
        crawler.start(word, args.total_page, args.start_page, args.per_page)
