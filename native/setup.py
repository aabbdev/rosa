from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extension = Pybind11Extension(
    "rosa_native_step",
    ["src/rosa_native_step.cpp"],
    cxx_std=17,
    define_macros=[("NDEBUG", "1")],
    extra_compile_args=["-O3"],
)

setup(
    ext_modules=[extension],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
