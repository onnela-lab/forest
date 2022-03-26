.. forest documentation master file, created by
   sphinx-quickstart on Thu Mar 24 19:57:29 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to forest's documentation!
==================================

Forest is a module used to analyze digital phenotyping data collected from the Beiwe app.

.. toctree::
   sycamore.md
   :maxdepth: 2
   :caption: Contents:

Each forest tree includes main functions which read in csv inputs and write csv files with results of analyses.

To analyze text/call data, use ``forest.willow.log_stats.log_stats_main``
-------------------------------------------------------------------------

.. automodule:: forest.willow.log_stats
   :members: log_stats_main
   :undoc-members:
   :show-inheritance:

To analyze GPS data, use ``forest.jasmine.traj2stats.gps_stats_main``
---------------------------------------------------------------------

.. automodule:: forest.jasmine.traj2stats
   :members: gps_stats_main
   :undoc-members:
   :show-inheritance:

To analyze survey data, use ``forest.sycamore.survey_stats_main``
-----------------------------------------------------------------

.. automodule:: forest.sycamore.base
   :members: survey_stats_main
   :undoc-members:
   :show-inheritance:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`







