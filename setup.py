import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "deepffr",
    version = "0.0.2",
    author = "Federico Lucchetti",
    author_email = "fedlucchetti@gmail.com",
    description = ("Deep FFR Utility."),
    license = "BSD",
    keywords = "example documentation tutorial",
    url = "https://github.com/fedlucchetti/DeepFFR",
    packages=find_packages(include=['deepffr']),
    package_data={"deepffr": ["models/*.h5"],},
    include_package_data=True,
    long_description=read('README.md'),
    # install_requires=[
    #         'tensorflow',
    #         'tqdm',
    #         'matplotlib'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
