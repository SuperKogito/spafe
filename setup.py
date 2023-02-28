import setuptools
from setuptools import setup, find_packages
from pathlib import Path
from spafe import version

here = Path(__file__).parent

# get readme text
with open(here / "README.md") as readme_file:
    LONG_DESCRIPTION = readme_file.read()

with open(here / "requirements.txt") as req_file:
    requirements = req_file.read().split()

setup(
    name="spafe",
    version=version.__version__,
    author="Ayoub Malek",
    author_email="superkogito@gmail.com",
    maintainer="Ayoub Malek",
    maintainer_email="superkogito@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/SuperKogito/spafe",
    license="BSD",
    description="Simplified Python Audio-Features Extraction.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="audio processing, features extraction",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Editors",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    platforms="any",
    extras_require={
        "tests": [
            "pytest>=6.2.4",
            "pytest-cov",
            "codecov",
            "coveralls",
            "pytest-xdist",
            "codacy-coverage",
            "matplotlib",
            "mock==4.0.3",
        ],
        "docs": [
            "sphinxcontrib-napoleon==0.7",
            "nbsphinx==0.8.9",
            "pydata-sphinx-theme==0.8.1",
            "matplotlib",
        ],
        "plotting": {"maplotlib"},
    },
)
