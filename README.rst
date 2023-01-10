covid_lit_contra_claims package README
======================================
A package for finding contradictory claims related to COVID-19 drug treatments in the CORD-19 literature

Installation
------------
To download this code and install in development mode, do the following:

.. code-block::

    $ cd {HOME}
    $ git clone https://github.com/dnsosa/covid_lit_contra_claims.git
    $ cd covid_lit_contra_claims
    $ pip install -e .


Usage
-----
This package is currently set up so that the training of the BERT model can be easily run as a package using a
command-line interface as follows:

.. code-block::

    $ # Make sure that installation was successful as described above
    $
    $ python -m covid_lit_contra_claims [OPTIONS]

Documentation about each argument and condition for running an experiment is provided using the ``--help`` flag.

.. code-block::

    $ python -m covid_lit_contra_claims --help
