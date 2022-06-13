import sys
from pathlib import Path
import os

import setuptools

long_description = (Path(__file__).parent / "README.md").read_text()

if sys.version_info < (3, 6):
    sys.exit("Python>=3.6 is required by ForteAug.")

setuptools.setup(
    name="forte.aug",
    version='0.1.0',
    url="https://github.com/asyml/ForteAug",
    description="NLP pipeline framework for data augmentation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License Version 2.0",
    packages=setuptools.find_namespace_packages(
        include=['fortex.forteaug', 'ftx.*'],
        exclude=["scripts*", "examples*", "tests*"]
    ),
    namespace_packages=["fortex"],
    include_package_data=True,
    install_requires=[
        'forte~=0.2.0',
        "sortedcontainers>=2.1.0",
        "numpy>=1.16.6",
        "jsonpickle>=1.4",
        "pyyaml>=5.4",
        "smart-open>=1.8.4",
        "typed_astunparse>=2.1.4",
        "funcsigs>=1.0.2",
        "typed_ast>=1.5.0",
        "jsonschema>=3.0.2",
        'typing>=3.7.4;python_version<"3.5"',
        "typing-inspect>=0.6.0",
        'dataclasses~=0.7;python_version<"3.7"',
        'importlib-resources>=5.1.4;python_version<"3.7"',
        "asyml-utilities",
        "texar-pytorch>=0.1.4",
        "tensorflow>=1.15.0",
        "requests",
    ],
    extras_require={
        "test": [
            "ddt",
            "testfixtures",
            "testbook",
            "termcolor",
            "transformers>=4.15.0",
            "nltk",
        ]
    },
)
