import os
from setuptools import setup


# __version__ = None

# abspath = os.path.abspath(os.path.dirname(__file__))
# exec(open(os.path.join(abspath, "lcurvetools/version.py")).read())

from lcurvetools import __version__

######################################################################################################
################ You May Remove All the Comments Once You Finish Modifying the Script ################
######################################################################################################

setup(
    name = 'lcurvetools', 
    version = __version__,
    description = 'Simple tools to plot learning curves of machine learning models created by scikit-learn or keras framework..',
    py_modules = ["lcurvetools"],
    package_dir = {'':'lcurvetools'},
    author = 'Andriy Konovalov',
    author_email = 'kandriy74@gmail.com',
    long_description = open('README.md').read() + '\n\n' + open('CHANGELOG.md').read(),
    long_description_content_type = "text/markdown",
    
    '''
    The url to where your package is stored for public view. Normally, it will be the github url to the repository you just forked.
    '''
    url='https://github.com/jinhangjiang/morethansentiments',
    
    '''
    Leave it as deafult.
    '''
    include_package_data=True,
    
    '''
    This is not a enssential part. It will not affect your package uploading process. 
    But it may affect the discoverability of your package on pypi.org
    Also, it serves as a meta description written by authors for users.
    Here is a full list of what you can put here:
    
    https://pypi.org/classifiers/
    
    '''
    classifiers  = [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: BSD License",
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Text Processing',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
    ],
    
    
    '''
    This part specifies all the dependencies of your package. 
    "~=" means the users will need a minimum version number of the dependecies to run the package.
    If you specify all the dependencies here, you do not need to write a requirements.txt separately like many others do.
    '''
    install_requires = [

        'pandas ~= 1.2.4',
        ...

    ],
    
    
    
    '''
    The keywords of your package. It will help users to find your package on pypi.org
    '''
    keywords = ['Text Mining', 'Data Science', ...],
    
)
