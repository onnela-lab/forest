from setuptools import setup, find_packages

requires = [
    'holidays',  # poplar
    'numpy',
    'openrouteservice',
    'pandas',
    'pyproj',  # jasmine
    'pytz',  # jasmine, poplar
    'ratelimit',
    'requests',  # bonsai
    'scipy',
    'shapely',  # jasmine
    'ssqueezepy',  # oak
    'timezonefinder',  # poplar, bonsai
    'wheel',  # for ratelimit
    'librosa'  # for audio file durations in sycamore
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
    packages=find_packages(include=["forest*"]),
    package_data=package_data,
    install_requires=requires
)
