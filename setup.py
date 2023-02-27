from setuptools import setup, find_packages

requires = [
    'holidays',  # poplar
    'librosa',  # for audio file durations in sycamore
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
    'wheel'  # for ratelimit
]

package_data = {'forest.poplar.raw': ['noncode/*.csv', 'noncode/*.json']}

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE.md', encoding='utf-8') as f:
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
