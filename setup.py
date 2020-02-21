from setuptools import setup, find_packages
import argparse
import sys

basic_install_requires = [
    "numpy",
    "matplotlib",
    "cython",
    "plotly==4.0.0", 
    "flask==1.0.2",
    "psutil",
]
ver_num = "0.1.0"

"""NOTE: You have to install tensorflow yourself, in case you are using
Tensorflow for your experiment"""
# tf_requires = ["tensorflow-cpu>=2.1.0"]
# tf_gpu_requires = ["tensorflow-gpu>=2.1.0"]
tfb_requires = ["tensorboard==2.0.2"]

setup(
    name= "exptools",
    version= ver_num + "tf",
    packages= find_packages(),
    install_requires= basic_install_requires+tfb_requires,
    license= "MIT License",
    long_dexcription= open("README.md").read(),
)
    
