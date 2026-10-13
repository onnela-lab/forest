# Overview

Information for contributors

## Quick start
In your terminal, navigate to your local copy of the Forest repo and...
* Install Forest in "editable" mode along with development libraries and typing stubs:
  * `pip install -e ".[dev]"`
  * `mypy --install-types --non-interactive`
* With stubs installed, you can then just run `mypy` confirm type checking will pass.
* To check code style and other linting details run:
  * `ruff check`
  * There are `ruff` extensions for the major IDEs, we recommend you use them.
* Run the test suite with `pytest`
  * _Or speed it up with `pytest -n 10` to run tests in parallel_
* To validate the citation file: `cffconvert -i CITATION.cff --validate`

Please see the [CONTRIBUTING.md](https://github.com/onnela-lab/forest/blob/develop/CONTRIBUTING.md)
file for more detailed information on contributing to Forest.

For a short start you can look at [our support page](https://github.com/onnela-lab/forest/blob/develop/SUPPORT.md)

### Documentation
Install required dependencies:
```shell
pip install -r docs/requirements.txt
```
Build the docs:
```shell
cd docs
make html
```
Open `docs/_build/html/index.html` in a web browser to check the results

## General information
* [Forest Naming Conventions](naming.md)
* [Exception handling](exceptions.md)
* [Logging](logging.md)
* Ask questions about software design before writing the code
* Delete unnecessary code (don't just comment it out)
* Rewrite code if necessary
* [Write short functions that do one thing only](https://github.com/amontalenti/elements-of-python-style#if-the-implementation-is-hard-to-explain-its-a-bad-idea)
* Specify exact version numbers for dependencies (pin your requirements)
* Refactor the code and avoid copy-paste programming ([Don't Repeat Yourself](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself))
* Write tests for all new code

## Python

There are many authors of Forest so style may not be consistent, particularly across Forest Trees.
We do _not_ use a code formatter tool like `black`, and we do not enforce PEP8.

* Target a code width of 100 characters, its ok to go up to 105, above that `ruff check` will error.
* Do not target code width of 80 characters. (PEP8 is deprecated for a reason.)
* Use descriptive variable and function names. Assume the reader of your code does not have the full
  context of how your code is used. Sometimes length is necessary for clarity, use discression.
* Except with `lambda`s, generator expressions, comprehensions, and when using `_` as for a
  throwaway variable, avoid one character names. Occasionally, like in well named pure math
  functions, one character names are acceptable.
* Use double quotes for strings and use single quotes only to avoid backslashes in the string
* Alphabetize imports within blocks. We have an isort configuration that you can use as well.
* Try to use parentheses for [continuations](https://github.com/amontalenti/elements-of-python-style#use-parens--for-continuations) and [method chaining](https://github.com/amontalenti/elements-of-python-style#use-parens--for-fluent-apis).
* Place whitespace before and after your operators (like "a | b", not "a|b")
* Use the [logging module](https://docs.python.org/3/library/logging.html) instead of `print()` in all final code.
* [Formatting log messages](http://reinout.vanrees.org/weblog/2015/06/05/logging-formatting.html): `logger.info("%s went %s wrong", 42, 'very')`
* Continue writing code and comments all the way until the end of the line then indent appropriately
* [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects)

### Packaging and distribution

#### Create a release
* Update the version number in `pyproject.toml` on the develop branch (for example: `1.1.0`)
* Commit and push all local changes to GitHub
* Merge the develop branch into the main branch
* Create a tag for the release: `git tag -a v1.1.0 -m "Release 1.1.0"`
* Push the tag to GitHub: `git push origin v1.1.0`
* [Create a release on GitHub](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) and auto-generate the changelog
* Build and upload the distribution archives to Python Package Index (see below) 

#### Upload to Python Package Index
* Use [TestPyPI](https://test.pypi.org/) for testing that your package can be uploaded, downloaded, and installed correctly
* [Register an account](https://test.pypi.org/account/register/)
* [Create an API token](https://test.pypi.org/manage/account/#api-tokens) (setting the "Scope" to "Entire account")
* [Add API token](https://packaging.python.org/en/latest/specifications/pypirc/#using-a-pypi-token) to your `$HOME/.pypirc` file
* Clear the build directory: `rm -r dist`
* Generate distribution archives: `python -m build`
* Check the results: `twine check dist/*`
* Upload distribution archives:
  * TestPyPI: `twine upload --repository testpypi dist/*`
  * PyPI: `twine upload dist/*`
* Install from TestPyPI to verify:
  * TestPyPI: `pip install --index-url https://test.pypi.org/simple/ --no-deps beiwe-forest`
  * PyPI: `pip install beiwe-forest`

## GitHub

Development workflow:
1. open an issue
1. create a pull request
1. merge pull request
1. close issue

### Branches

* Follow the [Gitflow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) for managing branches
* Use descriptive names for branches
* Keep branches short-lived to avoid potential merge conflicts

### Pull requests
* PRs:
  * are not just peer review but a way to communicate and learn about the code base
  * help build and pass on institutional knowledge, and keep default branch commit history clean
* Keep PRs small (ideally, under 100 lines of code) and self-contained (code + tests + docs) to make reviews easier and faster
* If you need to update an existing PR simply add commits to the corresponding feature branch instead of creating a new separate PR
* Make sure your PR has the latest changes from the develop branch and that it passes the [build process](https://github.com/onnela-lab/forest/actions/workflows/build.yml)
* Name your PR based on the task it serves to complete, not the changes that it implements.


#### Create a PR
1. Create a feature branch off the default branch: `git switch -c new-feature develop`
1. Push the new feature branch upstream to GitHub: `git push --set-upstream origin new-feature`
1. [Create a PR on GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request):
   * write a short description
   * [link to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) if applicable
   * select a reviewer to notify
   * add labels
1. Push commits to the upstream feature branch to update the PR or respond to reviewer's comments
1. [Re-request the review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) after each round of changes
1. After the PR is merged delete the feature branch in your local repository: `git branch -d new-feature`
1. You can see some further information about expectations for PRs on [our support document](https://github.com/onnela-lab/forest/blob/develop/SUPPORT.md).


#### Review a PR
1. [Comment on proposed changes](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/commenting-on-a-pull-request)
1. Resolve conversations when changes are addressed (either in code or comments)
1. [Approve the PR](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/approving-a-pull-request-with-required-reviews)
1. [Squash and merge](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/merging-a-pull-request) when all concerns are addressed and checks are completed
1. Delete the feature branch on GitHub

### Issues

* Use the bug report issue template to report bugs
* Assign to the right person
* Don't close issues until the fix is merged in to the default branch

## Trees

For maximum happiness of the end users, we would like to point you to [Forest naming conventions](naming.md)]. When analyzing digital phenotyping data, we recommend you to use the [logging guidelines](logging.md) to record what happens (...and what fails...) during analyses. No worries, we have prepared some code snippets for you to use.

### Tree naming

When you've finalized your contribution (hereafter 'tree'), you can decide on a name for your tree. Please keep our tree naming conventions and reserved trees in mind.

## External resources

* [Best Practices for Scientific Computing](https://doi.org/10.1371/journal.pbio.1001745)
* [Good enough practices in scientific computing](https://doi.org/10.1371/journal.pcbi.1005510)
* [Ten simple rules on writing clean and reliable open-source scientific software](https://doi.org/10.1371/journal.pcbi.1009481)
