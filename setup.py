import sys
from pathlib import Path
import os

import setuptools

long_description = (Path(__file__).parent / "README.md").read_text()

if sys.version_info < (3, 6):
    sys.exit("Python>=3.6 is required by ForteAug.")

setuptools.setup(
    name="ForteAug",
    version='0.0.1',
    url="https://github.com/asyml/ForteAug",
    description="A rich Data Augmentation library supporting structured NLP data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License Version 2.0",
    packages=setuptools.find_namespace_packages(
        include=['fortex.aug', 'ftx.*'],
        exclude=["scripts*", "examples*", "tests*"]
    ),
    namespace_packages=["fortex"],
    include_package_data=True,
    install_requires=[
        'forte~=0.2.0',
        "texar-pytorch>=0.1.4",
        "tensorflow>=1.15.0",
        "requests",
        "transformers>=4.15.0",
        "nltk",
    ],
    extras_require={
        "test": [
            "ddt",
            "testfixtures",
            "testbook",
            "termcolor",
        ]
    },
)
