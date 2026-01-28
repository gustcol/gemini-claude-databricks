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
    version="0.1.0",
    author="Guxxxta / Gustcol",
    author_email="gustcol@gmail.com",
    description="AI Assistant for Databricks using Gemini and Claude",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gustcol/gemini-claude-databricks",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "google-generativeai>=0.3.0",
        "anthropic>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "spark": [
            "pyspark>=3.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-assistant=ai_assistant.cli:main",
        ],
    },
)
