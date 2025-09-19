# setup.py  (place at project root)
from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->list[str]:
    """This function will return the list of requirements"""
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="customer_lifetime_value",          # package name used by pip install
    version="0.0.1",                         # bump each release
    author="Aryan Patial",
    author_email="aryanpatial31@example.com",
    description="End-to-end ML pipeline for customer lifetime value prediction.",
    packages=find_packages(where="src"),     # include everything inside src/
    install_requires= get_requirements('requirements.txt'), # external packages as dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
