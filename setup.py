from setuptools import setup
import glob
import os

setup(
    name="CARBS",
    version="0.0.1",
    author="CANUCS team",
    author_email="kartheik.iyer@dunlap.utoronto.ca",
    url = "https://github.com/NIRISS/CARBS",
    packages=["CARBS"],
    description="CANUCS Adaptive Resolved Bayesian SED-Fitting",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"], "CARBS": ["filters/*.*", "data/*.*"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        ],
    install_requires=["matplotlib", "numpy", "scipy", "george", "sklearn", "dense_basis", "grizli", "emcee"]
)
