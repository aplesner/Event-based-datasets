from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "load_data",
        ["load_data.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
]

setup(
    name='load_data',
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
)
