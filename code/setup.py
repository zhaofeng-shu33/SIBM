# currently used to compile
# the accelerated version of the simulation code
# use the command: python3 setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
setup(
    name='sibm_c',
    ext_modules=cythonize('sibm_c.pyx')
)