from setuptools import setup, find_packages

setup(
    name='weixianwei',  # 库的名称
    version='1.0.0',  # 版本号
    description='A library for weixianwei',  # 简要描述
    packages=find_packages(),  # 需要打包的目录列表
    author='weixianwei',  # 作者名字
    author_email='weixianwei0129@gmail.com',  # 作者邮箱
    url='https://github.com/weixianwei0129/weixianwei',  # 项目的URL
    classifiers=[  # 分类器列表，用于指定该项目的属性
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
