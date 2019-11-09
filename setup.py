# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


# get readme text
f = open('README.md', 'r')
try:
    long_desc = f.read()
finally:
    f.close()

# define requirements
requires = ["numpy>=1.17.2", "scipy>=1.3.1"]

setup (
        name         = 'spafe',
        version      = '0.1',
        author       = 'SuperKogito',
        author_email = 'superkogito@gmail.com',
        description  = 'Simplified python Audio Features Extraction',
        long_description = f,
        long_description_content_type = long_desc,
        license      = 'BSD',
        url          = 'https://github.com/SuperKogito/spafe',
        packages     = find_packages(),
        classifiers  = [
                        'Development Status :: 3 - Alpha',
                        'Environment :: Console',
                        'Environment :: Web Environment',
                        'Intended Audience :: Developers',
                        'License :: OSI Approved :: BSD License',
                        'Operating System :: OS Independent',
                        'Programming Language :: Python',
                        'Topic :: Documentation',
                        'Topic :: Utilities',
                      ],
        platforms            = 'any',
        include_package_data = True,
        install_requires     = requires,
     )
