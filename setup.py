"""
Setup script for gemini-claude-databricks package.

This package provides AI integration for Databricks using
Google Gemini and Anthropic Claude models.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gemini-claude-databricks",
    version="0.2.0",
    author="Guxxxta / Gustcol",
    author_email="gustcol@gmail.com",
    description="AI Assistant for Databricks using Gemini and Claude",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gustcol/gemini-claude-databricks",
    project_urls={
        "Bug Tracker": "https://github.com/gustcol/gemini-claude-databricks/issues",
        "Documentation": "https://github.com/gustcol/gemini-claude-databricks#readme",
        "Source Code": "https://github.com/gustcol/gemini-claude-databricks",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "notebooks"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: Databricks",
    ],
    keywords=[
        "databricks",
        "gemini",
        "claude",
        "ai",
        "llm",
        "spark",
        "delta-lake",
        "unity-catalog",
        "data-engineering",
        "anthropic",
        "google-ai",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "gemini": [
            "google-generativeai>=0.3.0",
        ],
        "claude": [
            "anthropic>=0.18.0",
        ],
        "all": [
            "google-generativeai>=0.3.0",
            "anthropic>=0.18.0",
        ],
        "spark": [
            "pyspark>=3.4.0",
        ],
        "mlflow": [
            "mlflow>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "ruff>=0.1.0",
        ],
        "full": [
            "google-generativeai>=0.3.0",
            "anthropic>=0.18.0",
            "pyspark>=3.4.0",
            "mlflow>=2.0.0",
        ],
    },
)
