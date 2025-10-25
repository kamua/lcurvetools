import os
from setuptools import setup
from setuptools import find_packages


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


if os.path.exists("lcurvetools/version.py"):
    VERSION = get_version("lcurvetools/version.py")
else:
    VERSION = get_version("lcurvetools/__init__.py")

setup(
    name="lcurvetools",
    version=VERSION,
    description=(
        "Simple tools for Python language to plot learning curves of a neural"
        " network model trained with the keras or scikit-learn framework."
    ),
    author="Andriy Konovalov",
    author_email="kandriy74@gmail.com",
    license="BSD 3-Clause License",
    long_description=open("README.md").read()
    + "\n\n"
    + open("CHANGELOG.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kamua/lcurvetools",
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.10",
    install_requires=["numpy", "matplotlib", "scikit-learn"],
    packages=find_packages(exclude=("test_*.py",)),
    keywords=[
        "learning curve",
        "keras history",
        "loss_curve",
        "validation_score",
    ],
)
