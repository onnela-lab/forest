from setuptools import setup, find_namespace_packages

requires = [
    'dataclasses',
    'numpy',
    'openrouteservice',
    'pandas',
    'scipy',
    'holidays',  # poplar
    'pytz',  # jasmine, poplar
    'timezonefinder',  # poplar, bonsai
    'requests',  # bonsai
]

package_data = {'': ['*.csv', '*.json']}

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='forest',
    version='0.1.1',
    description='--add description here--',
    long_description=readme,
    author='Onnela Lab',
    author_email='onnela.lab@gmail.com',
    license=license,
    packages=find_namespace_packages(include=['forest.*']),
    package_data=package_data,
    install_requires=requires
)
