# -*- coding: utf-8 -*-
import os
import codecs
import pathlib
from setuptools import setup, find_packages

################################################################################
# HELPER FUNCTIONS #############################################################
################################################################################
def get_lookup():
    """
    Get package information by way of spafe.version.

    Returns:
        (dict) : lookup dictionary with several global variables without
                 needing to import singularly.
    """
    lookup = dict()
    version_file = os.path.join("spafe", "version.py")
    with open(version_file) as fd:
        exec(fd.read(), lookup)
    return lookup


def get_reqs(lookup, key):
    """
    Get requirements and versions from
    the lookup obtained with get_lookup

    Returns:
        (list) : list of requirements.
    """
    install_requires = []
    for module in lookup[key]:
        module_name = module[0]
        module_meta = module[1]

        if "exact_version" in module_meta:
            dependency = "%s==%s" % (module_name, module_meta["exact_version"])

        elif "min_version" in module_meta:
            if module_meta["min_version"] == None:
                dependency = module_name

            else:
                dependency = "%s>=%s" % (module_name, module_meta["min_version"])

        install_requires.append(dependency)
    return install_requires


# Make sure everything is relative to setup.py
install_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(install_path)

# get readme text
with open("README.md") as filey:
    LONG_DESCRIPTION = filey.read()

# define infoss
lookup = get_lookup()
VERSION = lookup["__version__"]
NAME = lookup["NAME"]
AUTHOR = lookup["AUTHOR"]
AUTHOR_EMAIL = lookup["AUTHOR_EMAIL"]
PACKAGE_URL = lookup["PACKAGE_URL"]
KEYWORDS = lookup["KEYWORDS"]
DESCRIPTION = lookup["DESCRIPTION"]
LICENSE = lookup["LICENSE"]


if __name__ == "__main__":

    INSTALL_REQUIRES = get_reqs(lookup, "INSTALL_REQUIRES")
    TESTS_REQUIRES = get_reqs(lookup, "TESTS_REQUIRES")

    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=AUTHOR,
        maintainer_email=AUTHOR_EMAIL,
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        url=PACKAGE_URL,
        license=LICENSE,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        keywords=KEYWORDS,
        install_requires=INSTALL_REQUIRES,
        tests_require=TESTS_REQUIRES,
        extras_require={},
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            'Topic :: Multimedia :: Sound/Audio :: Editors',
            'Topic :: Multimedia :: Sound/Audio :: Speech',
            'Topic :: Scientific/Engineering :: Information Analysis',
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        platforms = 'any',
    )
