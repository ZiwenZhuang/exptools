from setuptools import setup, find_packages

setup(
    name= "exptools",
    version= "0.1.0dev",
    packages= find_packages(),
    install_requires= [
        "numpy",
        "matplotlib",
        "cython",
        "plotly==4.0.0", 
        "flask==1.0.2",
        "psutil",
        "imageio",
    ],
    license= "MIT License",
    long_dexcription= open("README.md").read(),
)
