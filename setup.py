from setuptools import setup, find_packages

setup(
    name='wxw',  # 库的名称
    version='0.1.2',  # 版本号
    keywords=['pip', 'wxw'],
    description='A library for wxw',  # 简要描述
    long_description="Includes some ways to work with pictures, add qt utils",
    author='weixianwei',  # 作者名字
    author_email='weixianwei0129@gmail.com',  # 作者邮箱
    url='https://github.com/weixianwei0129/weixianwei',  # 项目的URL

    packages=find_packages(),  # 需要打包的目录列表
    platforms="any",
    install_requires=["numpy", "opencv-python", "matplotlib"]  # 这个项目依赖的第三方库
)
