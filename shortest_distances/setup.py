from setuptools import setup, Extension
import numpy

setup(
    ext_modules=[
        Extension(
            "shortest_distances",
            ["src/shortest_distances.pyx"],
            include_dirs=[numpy.get_include()],
            language="c++",
        )
    ]
)
