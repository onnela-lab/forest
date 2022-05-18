# Overview

Information for contributors

## Quick start
* Change current directory to the top level of your local Forest repository
* Install Forest in editable mode: `pip install -e .`
* Install development tools: `pip install -r requirements.txt`
* Run code style checks: `flake8`
* Run type hint checks: `mypy -p forest`
* Run test suite: `pytest`

### Documentation
Install required dependencies:
```shell
pip install docs/requirements.txt
```
Build the docs:
```shell
cd docs/source
make html
```
Open `docs/_build/html/index.html` in a web browser to check the results

## General information
* [Forest Naming Conventions](naming.md)
* [Exception handling](exceptions.md)
* [Logging](logging.md)
* Delete unnecessary code (don't just comment it out)
* [Write short functions that do one thing only](https://github.com/amontalenti/elements-of-python-style#if-the-implementation-is-hard-to-explain-its-a-bad-idea)
* Specify exact version numbers for dependencies (pin your requirements)
* Refactor the code and avoid copy-paste programming ([Don't Repeat Yourself](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself))
* Write tests for all new code

## Python
* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/), [The Zen of Python](https://www.python.org/dev/peps/pep-0020/), [Google style guide for docstrings](https://google.github.io/styleguide/pyguide.html#s3.8.1-comments-in-doc-strings), and [The Elements of Python Style](https://github.com/amontalenti/elements-of-python-style) guidelines
* Avoid one character names (except with `lambda`, generator expressions, and with unpacking when using `_` as a throwaway character)
* Use double quotes for strings and use single quotes only to avoid backslashes in the string
* Alphabetize imports within blocks
* Use parentheses for [continuations](https://github.com/amontalenti/elements-of-python-style#use-parens--for-continuations) and [method chaining](https://github.com/amontalenti/elements-of-python-style#use-parens--for-fluent-apis)
* Put a line break before a binary operator
* Use the [logging module](https://docs.python.org/3/library/logging.html) instead of `print()`
* [Formatting log messages](http://reinout.vanrees.org/weblog/2015/06/05/logging-formatting.html): `logger.info("%s went %s wrong", 42, 'very')`
* Continue writing code and comments all the way until the end of the line then indent appropriately

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
* Keep PRs small (ideally, under 100 lines of code) to help makes reviews easier and faster
* If you need to update an existing PR simply add commits to the corresponding feature branch instead of creating a new separate PR

How to create a PR:
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

How to review PR:
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

For maximum happiness of the end users, we would like to point you to [[forest naming conventions | Forest Naming Conventions]]. When analyzing digital phenotyping data, we recommend you to use the [[the logging module|Logging Guidelines]] to record what happens (...and what fails...) during analyses. No worries, we have prepared some code snippets for you to use.

### Tree naming

When you've finalized your contribution (hereafter 'tree'), you can decide on a name for your tree. Please keep our [[tree naming conventions and reserved trees|Tree Naming conventions]] in mind.

## External resources

* [Best Practices for Scientific Computing](https://doi.org/10.1371/journal.pbio.1001745)
* [Good enough practices in scientific computing](https://doi.org/10.1371/journal.pcbi.1005510)
* [Ten simple rules on writing clean and reliable open-source scientific software](https://doi.org/10.1371/journal.pcbi.1009481)
