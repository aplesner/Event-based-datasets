from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Get the directory containing this setup.py file
setup_dir = os.path.dirname(os.path.abspath(__file__))

extensions = [
    Extension(
        "load_data",
        [os.path.join(setup_dir, "load_data.pyx")],
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
