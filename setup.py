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

# NOTE: the version is not check here
tf_requires = ["tensorflow>=2.1.0"]
tf_gpu_requires = ["tensorflow-gpu>=2.1.0"]
tfb_requires = ["tensorboard>=2.1.0"]

parser = argparse.ArgumentParser()
parser.add_argument(
    '--tfb', help= 'An option to choose whether to install tensorboard as viewing method: 0 for no tfb, 1 for tensorflow, 2 for tensorflow-gpu',
    type= int, default= 0,
)
args = parser.parse_args()

if args.tfb == 0:
    setup(
        name= "exptools",
        version= ver_num + "base",
        packages= find_packages(),
        install_requires= basic_install_requires,
        license= "MIT License",
        long_dexcription= open("README.md").read(),
    )
elif args.tfb == 1:
    setup(
        name= "exptools",
        version= ver_num + "tf",
        packages= find_packages(),
        install_requires= basic_install_requires+tf_requires+tfb_requires,
        license= "MIT License",
        long_dexcription= open("README.md").read(),
    )
elif args.tfb == 2:
    setup(
        name= "exptools",
        version= ver_num + "tfgpu",
        packages= find_packages(),
        install_requires= basic_install_requires+tf_gpu_requires+tfb_requires,
        license= "MIT License",
        long_dexcription= open("README.md").read(),
    )
    
