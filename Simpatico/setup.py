from setuptools import setup, find_packages

setup(
    name="simpatico",
    version="0.1",
    packages=find_packages(include=["simpatico", "simpatico.*"]),
    include_package_data=True,
    package_data={
        "simpatico.data": ["*.json"],
        "simpatico.models.molecule_encoder": ["*.json"],
        "simpatico.models.protein_encoder": ["*.json"],
        "simpatico.models.weights": ["*.pt"],
    },
    entry_points={
        "console_scripts": [
            "simpatico = simpatico.cli:main",
        ],
    },
    install_requires=[
        "molvs==0.1.1",
        "rdkit==2024.9.6",
    ],
)
