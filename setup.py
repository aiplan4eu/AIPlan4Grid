# Always prefer setuptools over distutils
# To use a consistent encoding
from codecs import open
from os import path
from os.path import join as pjoin
from setuptools import setup
import plan4grid

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="plan4grid",
    version=plan4grid.__version__,
    description="AIPlan4Grid: a Python library for the AIPlan4EU project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aiplan4eu/AIPlan4Grid",
    author="Martin Debouté",
    author_email="martin.deboute@artelys.com",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    packages=["plan4grid"],
    include_package_data=True,
    install_requires=open(pjoin(HERE, "requirements", "requirements.txt")).read().splitlines(),
)
