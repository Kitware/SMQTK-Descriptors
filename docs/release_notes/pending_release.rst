Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Add workflow to inherit the smqtk-core publish workflow.

Miscellaneous

* Added a wrapper script to pull the versioning/changelog update helper from
  smqtk-core to use here without duplication.

Misc.

* Add PyTorch descriptor generator implementation

Testing

* Updated pytest configuration to cover package + tests, add report output
  options.

* Removed or no-cover mark dead lines of code.

Documentation

* Updated CONTRIBUTING.md to reference smqtk-core's CONTRIBUTING.md file.

Fixes
-----

CI

* Also run CI unittests for PRs targetting branches that match the `release*`
  glob.

Dependency Versions

* Update the locked version of urllib3 to address a security vulnerability.

* Update the locked version of pillow to address a security vulnerability.

* Update the developer dependency and locked version of ipython to address a
  security vulnerability.

* Removed `jedi = "^0.17.2"` requirement since recent `ipython = "^7.17.3"`
  update appropriately addresses the dependency.
