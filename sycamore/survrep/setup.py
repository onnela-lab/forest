from setuptools import setup, find_packages

requires = ['numpy', 'pandas']

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='survrep',
    version='0.0.1',
    description='Modules for processing raw Beiwe survey timings files',
    long_description=readme,
    author='Josh Barback',
    author_email='onnela.lab@gmail.com',
    license=license,
    packages=find_packages(),
    package_data = {'': ['*.json']},
    install_requires = requires
)
