from setuptools import setup, find_packages

setup(
    name='my_package',
    version='1.0',
    packages=find_packages(exclude=['python.lib.python3.12.site-packages.clang']),
    # other setup configurations...
)
