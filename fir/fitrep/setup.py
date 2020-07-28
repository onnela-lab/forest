from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='fitrep',
    version='0.0.1',
    description='Modules for working with Fitabase data',
    long_description=readme,
    author='Josh Barback',
    author_email='onnela.lab@gmail.com',
    license=license,
    packages=find_packages()
)
