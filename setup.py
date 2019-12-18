from setuptools import setup, find_packages

setup(
    name= "exptools",
    version= "0.1",
    packages= find_packages(),
    install_requires= [
        "numpy",
        "plotly==4.0.0", 
        "flask==1.0.2"
    ],
    license= "MIT License",
    long_dexcription= open("README.md").read(),
)
