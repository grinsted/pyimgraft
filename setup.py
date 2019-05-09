from setuptools import setup, find_packages

with open('requirements.txt') as fp: # Assume that we have a hand-crafted minimal requirements file should be minimal
    install_requires = fp.read()

setup(name="PYIMGRAFT", packages=find_packages(), install_requires=install_requires)