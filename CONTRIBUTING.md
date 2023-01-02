Contributing guidelines
=======================

How to contribute
-----------------

The preferred way to contribute to spafe is to fork the [main repository](https://github.com/SuperKogito/spafe) on GitHub:

1.	Fork the [project repository](https://github.com/SuperKogito/spafe): click on the 'Fork' button near the top of the page. This creates a copy of the code under your account on the GitHub server.

2.	Clone this copy to your local disk:

	-	Using SSH:

	```bash
	git clone git@github.com:YourLogin/spafe.git
	cd spafe
	```

	-	Using HTTPS&#x3A;

	```bash
	git clone https://github.com/YourLogin/spafe.git
	cd spafe
	```

3.	Remove any previously installed spafe versions, then install your local copy with testing dependencies:

	```bash
	pip uninstall spafe
	pip install .
	```

4.	Create a branch to hold your changes:

	```bash
	git checkout -b my-feature
	```

5.	Start making changes.

	```diff
	-> Please never work directly on the `master` branch!
	```

6.	Once you are done, make sure to format the code using black to fit spafe's codestyle.

	```black spafe/```

7.	Make sure that the tests succeed and have enough coverage.

	```pytest -x --cov-report term-missing --cov=spafe spafe/tests/test_*.py ```

8.	Use Git for the to do the version controlling of this copy. When you're done editing, you know the drill `add`, `commit` then `push`:

	```bash
	git add modified_files
	git commit
	```

	to record your changes in Git, push them to GitHub with:

	```bash
	git push -u origin my-feature
	```

9.	Finally, go to the web page of the your spafe fork repo, and click 'Pull request' button to send your changes to the maintainers to review.

Remarks
-------

It is recommended to check that your contribution complies with the following rules before submitting a pull request:

-	All public methods should have informative docstrings with sample usage presented.

	You can also check for common programming errors with the following tools:

-	Code with good test coverage (at least 80%), check with:

	```bash
	pytest
	```

-	Check code formatting using black:

	```bash
  black --check spafe
	```

Filing bugs
-----------

we use Github issues to track all bugs and feature requests. In the case of coming across a bug, having a question or a feature suggestion etc. please feel free to open an issue. However, please make sure that your issue comply with our rules/follows the provided templates:

-	[bug reports template](https://github.com/SuperKogito/spafe/blob/master/.github/ISSUE_TEMPLATE/bug_report.md)

-	[feature requests template](https://github.com/SuperKogito/spafe/blob/master/.github/ISSUE_TEMPLATE/feature_request.md)

	In addition, please check that your issue complies with the following rules before submitting:

-	Verify that your issue is not being currently addressed by other in[issues](https://github.com/SuperKogito/spafe/issues) or [pull requests](https://github.com/SuperKogito/spafe/pulls).

-	Please ensure all code snippets and error messages are formatted appropriately. See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-	Please include your operating system type and version number, as well as your Python, spafe, numpy, and scipy versions. This information can be found by running the following code snippet:

	```python
	import sys
	import numpy
	import scipy
	import spafe
	import platform

	print(platform.platform())
	print("Python", sys.version)
	print("NumPy", numpy.__version__)
	print("SciPy", scipy.__version__)
	print("spafe", spafe.__version__)
	```

Documentation
-------------

You can edit the documentation using any text editor and then generate the HTML output by typing `make html` from the docs/ directory. The resulting HTML files will be placed in `_build/html/` and are viewable in a web browser. See the README file in the doc/ directory for more information.

To build the documentation, you will need:

-	[sphinx](http://sphinx.pocoo.org/).
-	[sphinxcontrib-napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/).

Note
----

This document was based on the [scikit-learn](http://scikit-learn.org/) & [librosa](https://github.com/librosa/librosa) contribution guides.
