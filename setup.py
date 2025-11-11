#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from setuptools import Distribution, setup
from setuptools.extension import Extension

# Fetch numpy and Cython as build dependencies.
Distribution().fetch_build_eggs(["numpy", "Cython"])

# Safe to import here after line above.
import numpy  # noqa E402
from Cython.Build import cythonize  # noqa E402


setup(
    ext_modules=cythonize(
        [
            Extension(
                "metro.devices.analyze._seq_cov_native",
                ["src/metro/devices/analyze/_seq_cov_native.pyx"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                "metro.devices.display._fast_plot_native",
                ["src/metro/devices/display/_fast_plot_native.pyx"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                "metro.devices.display._hist2d_native",
                ["src/metro/devices/display/_hist2d_native.pyx"],
                include_dirs=[numpy.get_include()],
            ),
        ],
        language_level=3,
        build_dir="build",
    ),
)
