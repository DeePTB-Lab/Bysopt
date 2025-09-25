from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantum-well-optimization",
    version="1.0.0",
    author="",
    author_email="",
    description="Bayesian Optimization for Quantum Well Design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeePTB-Lab/Bysopt",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "quantum-well-opt=run_opt:main",
            "qwo=run_opt:main",
        ],
    },
    include_package_data=True,
)