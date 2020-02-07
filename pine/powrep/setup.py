from setuptools import setup, find_packages

requires = []

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='powrep',
    version='0.0.1',
    description='Modules for processing raw Beiwe power state data',
    long_description=readme,
    author='Josh Barback',
    author_email='onnela.lab@gmail.com',
    license=license,
    packages=find_packages(),
    install_requires = requires
)
