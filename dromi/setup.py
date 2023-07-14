#!/usr/bin/env python

"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib
from os.path import splitext
from os.path import basename
from glob import glob
here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the TOREAD_gleipnir.md file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(name='dromi',
      version='0.0.11',
      # list folders, not files
      packages=find_packages('src'),
      package_dir={'': 'src'},
      py_modules=[splitext(basename(path))[0] for path in glob('dromi/src/dromi/*.py')],
      #scripts=['bin/script1.py'],
      package_data={'dromi': ['data/*']},
      description= '',
      long_description=long_description,
      classifiers=[  # Optional
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 4 - Beta',
            # Pick your license as you wish
            'License :: OSI Approved :: MIT License',
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate you support Python 3. These classifiers are *not*
            # checked by 'pip install'. See instead 'python_requires' below.
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3 :: Only',
      ],
      python_requires='>=3.7, <4',
      project_urls={
            'Changelog': 'https://github.com/LysSanzMoreta/dromi/blob/master/CHANGELOG.rst',
            'Issue Tracker': 'https://github.com/LysSanzMoreta/dromi/issues',
      },
      install_requires=[
            'biopython>1.78',
            'pandas>1.5.2',
            'matplotlib>3.3.4',
            'dill>0.3.3',
            'seaborn>=0.11.2',
            'scipy>1.5.4',
            'scikit-learn>0.24.1',
      ],
      setup_requires=[
            'biopython>1.78',
            'pandas>1.5.2',
            'matplotlib>3.4.0',
            'seaborn>=0.11.2',
            'scipy>=1.5.4',
            'scikit-learn>0.24.1',
            'umap-learn>0.5.2',
      ],
      include_package_data=True,
      zip_safe=False

      )