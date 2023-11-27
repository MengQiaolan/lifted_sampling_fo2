import os
from setuptools import find_packages, setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='sampling_fo2',
    version='0.1',
    url='https://github.com/lucienwang1009/lifted_sampling_fo2',
    author='Lucien Wang',
    author_email='lucienwang@buaa.edu.cn',
    license='MIT',

    packages=find_packages(include=['sampling_fo2', 'sampling_fo2.*']),
    install_requires=["symengine",
                      "sympy",
                      "lark",
                      "numpy",
                      "networkx",
                      "contexttimer",
                      "logzero",
                      "pandas",
                      "pysat",
                      "tqdm",
                      "dataclasses",
                      "PrettyPrintTree"],
    python_requires='>=3.8',
    description = ("A exact model sampler (counter) for two-variables first-order logic."),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="sampling first-order-logic combinatorics lifted-inference",
)
