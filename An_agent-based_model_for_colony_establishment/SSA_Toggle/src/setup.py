from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import platform
import os

if platform.system() == 'Linux':
    os.environ["CC"] = "gcc"
    ext_modules = [Extension("toggle", ["toggle.pyx", "SSAToggle.cpp"], language='c++',
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=['-fopenmp', '-lstdc++'],
                             extra_link_args=['-fopenmp']
                             )]
else:
    ext_modules = [Extension("toggle", ["toggle.pyx", "SSAToggle.cpp"], language='c++',
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=["/openmp"]
                             )]
# if we use gcc/g++ in linux , extra_compile_args should be extra_compile_args=['-O3', '-fopenmp']
# if we use Windows extra_compile_args=["/openmp"]

setup(cmdclass={'build_ext': build_ext}, ext_modules=cythonize(ext_modules))
