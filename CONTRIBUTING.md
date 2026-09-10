# Contributing to Forest

Thanks you for your interest in contributing.

Forest is developed and maintained by the [Onnela Lab](https://www.hsph.harvard.edu/onnela-lab/) in
the Department of Biostatistics at the Harvard T.H. Chan School of Public Health, alongside the
[Beiwe platform](https://github.com/onnela-lab/beiwe-backend).

## Code of conduct

By participating in this project you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

Follow the directions for running the latest development version of the Forest package as your
local, editable copy of the package. Please report any difficulties you encounter so that we can
make our ReadMe directions as effective as possible.

If you are interested in more active development, running your work directly off our Forest repo
instead of a fork, or any kind of maintenance or issue moderatuion role, please reach out to us.

## Basics and Branches

The `develop` branch is where we stage any updates we think are ready for public use or review, and
potentially ready to push out to PyPI. `develop` is the default branch and will be the state if you
clone the repo. Do not create pull requests against `main`, that is our release branch.

## Ways to contribute

Pretty much everything has to go through [the issue
tracker](https://github.com/onnela-lab/forest/issues). You will need to fork the repo to submit your
own changes via pull requests if you work on a change yourself. There are many tutorials online to
do this, and you can ask an LLM too.

- **Comment or ask questions on the issue tracker.** We can only engage with you if you engage with
  us.
- **Report a bug.** Check out the [SUPPORT](SUPPORT.md) page for details on how to report bugs.
- **Optimization.** We might not know if some operation takes too long or needs too much memory for
  your use case, and performance is a heck of a feature. If you have a problem or an idea, open an
  issue describing it.
- **Submit a change directly.** Small fixes can go straight to a pull request. For larger changes,
  especially a new tree or analysis, please open an issue first so we can discuss the design before
  you invest the effort. (We may need to bring in that pull request on a new Onnela-Lab repo Forest
  branch, which might close that pull request in a way where it cannot be reopened.)
- **Improve documentation.** Good documentation is _hard._ If you notice gaps or errors in the
  documentation, its all contained in the docs/sources folder and all you need to know is markdown.
  We will probably accept any improvement you make.
- **Let us know about your use-case.** Share details about how you are using Forest, let us know
  about specific requirements you have, we will try to help you out.
- **Ask what we are working on.** We just need to know you want to help. Find the email address of
  our maintainer, its not hidden, and ask.
- **Request a feature or inform us of a new method.** Open an issue describing the analysis you have
  an interest in and the published method it is based on. (We are an academic group and interested
  in incorporating scientifically sound methods.)

## Adding a new tree

Forest is organized into independent subpackages called trees, each implementing one methodological
pipeline. A new tree should implement a method that has been described in the peer-reviewed
literature, use the shared data structures and conventions provided by the Poplar utility layer, and
ship with tests and documentation.

## Pull request expectations

- Branch from `develop` and keep the change focused on a single concern. We will guide you from there.
- Include tests covering new or changed behavior.
- Update the documentation under `docs/` when you change user-facing behavior.
- Code should pass the repository's linting and type checking, and follow the import conventions
  used by the surrounding subpackage. (these may not be fully consistent.)
- Describe what the change does and why in the pull request body in clear language.
- Provide authorship and academic credit information at the top of the file. See some existing
  instances in the repository for reference.
- LLM-generated contributions must be clearly indicated, and you must include information about
  _both the model and platform used, and the nature of its contribution_ in the code and in the pull
  request. Pull requests with LLM contributions must be heavily reviewed by you and other people
  before we will merge them.
  
# Academic contributions:
- Be Patient. We cannot rush this process.
- This process will be facilitated by the Forest maintainer(s).
- A maintainer has to review your work for _code quality_ first. All code requires maintenance,
  and pull requests must be kept fresh with other work in the repository. 
- We need to identify someone with the appropriate background and expertise to review your
  contribution, and they need to find the time to review it.
- It may help to get in contact with us directly. The human factor always matters, and we need to
  ensure academic credit is properly assigned.
- We also periodically interface with JOSS (Journal of Open Source Software), which reviews the
  content of this repo, and may have their own thoughts or processes.
