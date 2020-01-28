from setuptools import setup, find_namespace_packages

requires = ['holidays',
            'humanize',
            'pandas',
            'pytz', 
            'seaborn',
            'timezonefinder']

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='beiwetools',
    version='0.0.1',
    description='Classes and functions for working with Beiwe data sets and study configurations',
    long_description=readme,
    author='Josh Barback',
    author_email='onnela.lab@gmail.com',
    license=license,
    packages=find_namespace_packages(include = ['beiwetools.*']),
    package_data = {'': ['*.json']},
    install_requires = requires
)
