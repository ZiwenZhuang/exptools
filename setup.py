from setuptools import setup, find_packages

setup(
    name= "exptools",
    version= "0.1.0dev",
    packages= find_packages(),
    install_requires= [
        "numpy",
    ],
    license= "MIT License",
    long_dexcription= open("README.md").read(),
)