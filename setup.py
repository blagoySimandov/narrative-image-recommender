from setuptools import setup, find_packages

setup(
    name="narrative-image-recommender",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pydantic",
        "torch",
        "transformers",
        "pillow",
        "matplotlib",
    ],
    extras_require={
        "notebook": [
            "jupytext",
            "jupyter",
        ],
        "dev": [
            "datamodel-code-generator",
        ],
    },
    python_requires=">=3.9",
)
