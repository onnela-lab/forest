from setuptools import setup, find_namespace_packages

requires = ['numpy',
            'pandas',
            'scipy',
            'holidays', # poplar
            'pytz', # jasmine, poplar
            'timezonefinder', # poplar
            ]

package_data = {'': ['*.csv', '*.json']}

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name = 'forest',
    version = '0.1.1',
    description = 'Library to analyze smartphone health data (surveys and sensor data) from Beiwe and other platforms',
    long_description = readme,
    author = 'Onnela Lab',
    author_email = 'onnela.lab@gmail.com',
    license = license,
    packages = find_namespace_packages(include = ['forest.*']),
    package_data = package_data,
    install_requires = requires
)
