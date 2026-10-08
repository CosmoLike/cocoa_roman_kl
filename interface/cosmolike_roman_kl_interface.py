"""Loader stub of the compiled cosmolike interface of roman_kl.

cosmolike_roman_kl_interface.so, built in this folder from interface.cpp
by MakefileCosmolike (run by scripts/compile_roman_kl.sh), is a pybind11
extension module: a shared library that Python imports like a module,
which exposes cosmolike's C/C++ functions (init_*, set_*, compute_*) to
the likelihood, the tests and the notebooks. scripts/start_roman_kl.sh
puts this folder on PYTHONPATH.

This file is the stub setuptools writes next to such a library:
importing it runs __bootstrap__, which locates the .so beside this file
and loads it under the same module name. In a normal setup the stub does
not run: within one folder Python's import system tries extension
modules before .py files, so `import cosmolike_roman_kl_interface` loads
the .so directly, and without the .so the stub has nothing to load. It
also uses the imp module, which Python 3.12 removed (the Cocoa
environment runs Python 3.11).
"""

def __bootstrap__():
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'cosmolike_roman_kl_interface.so')
   __loader__ = None; del __bootstrap__, __loader__
   imp.load_dynamic(__name__,__file__)
__bootstrap__()
