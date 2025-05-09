from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='acdctools',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=install_requires
)
