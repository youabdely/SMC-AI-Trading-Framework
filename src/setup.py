from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        "strategy.pyx",
        compiler_directives={'language_level': "3"} # Fuerza el uso de la sintaxis de Python 3
    )
)