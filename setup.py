from setuptools import setup, find_packages

VERSION = "0.0.6"
DESCRIPTION = "Aipaca Neural Nets Training Time Predictor"
LONG_DESCRIPTION = "Very fast model training time predictor. (Currently supports Feed forward and Conv Nets)"

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="aipaca_predictor",
    version=VERSION,
    author="Cody Wang",
    author_email="codyw@aipaca-corp.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)
