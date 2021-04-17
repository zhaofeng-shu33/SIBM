from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
root_dir = '/usr/include/'
extensions = [
    Extension("sdp_admm_py", ["sdp_admm_py.pyx", "sdp_admm.cpp"],
              include_dirs=[root_dir + 'eigen3']
            )
    ]
setup(
    name='sdp_admm_py',
    ext_modules=cythonize(extensions),
    version = '0.3',
    author = 'zhaofeng-shu33',
    author_email = '616545598@qq.com',
    license = 'Apache License Version 2.0',
    description = 'SBM community detection with semi-definite programming'
)


