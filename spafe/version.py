"""

- Description : Version file.
- Copyright (c) 2019-2022 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
__version__ = "0.2.0"
AUTHOR = "Ayoub Malek"
AUTHOR_EMAIL = "superkogito@gmail.com"
NAME = "spafe"
PACKAGE_URL = "https://github.com/SuperKogito/spafe"
KEYWORDS = "audio processing, features extraction"
DESCRIPTION = "Simplified Python Audio-Features Extraction."
LICENSE = "BSD"


# Global requirements

INSTALL_REQUIRES = (
    ("numpy", {"min_version": "1.18.1"}),
    ("scipy", {"min_version": "1.4.1"}),
)

TESTS_REQUIRES = (("pytest", {"min_version": "6.2.4"}),)

INSTALL_REQUIRES_ALL = INSTALL_REQUIRES + TESTS_REQUIRES
