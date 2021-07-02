
from setuptools import setup, find_packages

from bcn import __version__

with open("README.md", "r") as fh:
   long_description = fh.read()

setup(
   name="bcn",
   version=__version__,
   description="Thesis code for testing branched connection networks, 2021.",
   license="MIT",
   packages=find_packages(),
   classifiers=[ # https://pypi.org/classifiers/
      "Development Status :: 3 - Alpha",
      "Programming Language :: Python :: 3.7",
      "Programming Language :: Python :: 3.8",
      "Programming Language :: Python :: 3.9",
      #"Programming Language :: Python :: 3.10",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
      "Intended Audience :: Science/Research",
      "Natural Language :: English",
      "Topic :: Scientific/Engineering :: Artificial Intelligence",
   ],
   long_description=long_description,
   long_description_content_type="text/markdown",
   install_requires=[
      #"matplotlib>=3.4.2",
      "numpy>=1.20.3",
      "PyQt5>=5.15.4",
      "scikit_learn>=0.24.2",
      "tqdm>=4.61.1",
   ],
   extras_require = {
      "dev": [
         "Sphinx==3.5.4",
         "sphinx-autodoc-typehints==1.12.0",
         "sphinx-book-theme==0.1.0",
         "mypy==0.910",
         "pytest==6.2.4",
         "check-manifest==0.46",
         "twine==3.4.1",
         "wheel>=0.36.2",
         "pipreqs==0.4.10",
      ]
   },
   python_requires=">=3.7.0",
   url="https://github.com/almonds0166/BCN",
   author="Madison Landry",
   author_email="mlandry@mit.edu",
   project_urls={
     "Documentation": "https://web.mit.edu/almonds/www/BCN/index.html",
     "Issue tracker": "https://github.com/almonds0166/BCN/issues",
   },
)