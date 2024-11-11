from setuptools import setup, find_packages

setup(
    name='wxw',
    version='1.0.4',
    keywords=['pip', 'wxw'],
    description='A library for wxw',
    long_description="Includes some ways to work with pictures, add qt utils",
    author='weixianwei',
    author_email='weixianwei0129@gmail.com',
    url='https://github.com/weixianwei0129/mylibrary',

    packages=find_packages(),
    platforms="any",
    install_requires=["numpy<2.0.0", "psutil", "opencv-python", "matplotlib", "Pillow", "einops", "PyYaml"]
)
