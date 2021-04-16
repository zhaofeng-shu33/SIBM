from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
if sys.path[-1].find('conda') > 0:
    conda_root = '/home/feng/.conda/envs/r_env/'
    root_dir = conda_root + 'lib/R/library/'
    r_dir = conda_root + 'lib/R/include/'
    r_so_dir = conda_root + 'lib/R/lib/'
else: # debian buster
    root_dir = '/usr/lib/R/site-library/'
    r_dir = '/usr/share/R/include/'
    r_so_dir = ''
extensions = [
    Extension("sdp_admm_py", ["sdp_admm_py.pyx", "sdp_admm.cpp"],
        include_dirs=[root_dir + 'RcppArmadillo/include',
                      root_dir + 'Rcpp/include',
                      root_dir + 'RInside/include',
                      r_dir],
        libraries = ['R', 'RInside', ':Rcpp.so', ':RcppArmadillo.so'],
        library_dirs = [root_dir + 'Rcpp/libs/',
            root_dir + 'RcppArmadillo/libs/',
            root_dir + 'RInside/lib/', r_so_dir])]
setup(
    name='sdp_admm_py',
    ext_modules=cythonize(extensions),
    version = '0.3',
    author = 'zhaofeng-shu33',
    author_email = '616545598@qq.com',
    license = 'Apache License Version 2.0',
    description = 'SBM community detection with semi-definite programming'
)


