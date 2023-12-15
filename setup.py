import os
from setuptools import setup
from setuptools import find_packages


# __version__ = None

abspath = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(abspath, "lcurvetools", "version.py")).read())

setup(
    name = 'lcurvetools', 
    version = __version__,
    description = 'Simple tools to plot learning curves of neural network models created by scikit-learn or keras framework.',
    author = 'Andriy Konovalov',
    author_email = 'kandriy74@gmail.com',
    license='BSD 3-Clause License',
    long_description = open('README.md').read() + '\n\n' + open('CHANGELOG.md').read(),
    long_description_content_type = "text/markdown",
    url='https://github.com/kamua/lcurvetools',
    include_package_data=True,
    classifiers  = [
        'Development Status :: 5 - Production/Stable',
        "License :: OSI Approved :: BSD License",
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",

    ],
    install_requires = [
        'numpy',
        'matplotlib'
    ],
    packages=find_packages(exclude=("*_test.py",)),
    keywords = ['learning curve', 'keras history', 'loss_curve', 'validation_score']
)
