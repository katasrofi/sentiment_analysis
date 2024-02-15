from setuptools import setup, Extension
from Cython.Build import cythonize

ext = [Extension("~/Document/Files/Documents/Files/MyEnv/lib/python3.10/site-packages/pandas/_libs", sources=["_libs/util.pxd"]),
       Extension("Module.tokenize", sources=["Module/tokenize.pyx"]),
       #Extension("Module")
       Extension("model", sources=["model.pyx"])]

setup(name="Model",
      ext_modules=cythonize(ext))