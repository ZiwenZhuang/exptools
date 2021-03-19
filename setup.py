from setuptools import setup, find_packages
import argparse
import sys

basic_install_requires = [
    "numpy",
    "matplotlib",
    "cython",
    "plotly>=4.0.0", 
    "flask>=1.0.2",
    "psutil",
    "pandas",
    "imageio",
]
ver_num = "2.0"

tfb_requires = ["tensorboard", "tensorboardX", "crc32c", "soundfile"]

setup(
    name= "exptools",
    version= ver_num,
    packages= find_packages(),
    install_requires= basic_install_requires+tfb_requires,
    license= "MIT License",
    long_dexcription= open("README.md").read(),
)

