"""Jazz AI Experiments."""

from setuptools import setup, find_packages

setup(
    name="jazzaiexperiments",
    version="0.0.1",
    description="Jazz AI experiments",
    author="David Su",
    packages=find_packages(),
    install_requires=[
        "h5py>=2.7.0",
        "keras>=2.0.5",
        "matplotlib>=2.0.0",
        "mido>=1.2.8",
        "numpy>=1.12.0"
    ]
)
