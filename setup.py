"""
Setup script for the retrieval package
"""

from setuptools import setup, find_packages

# 패키지 정보 읽기
with open("retrieval/__init__.py", "r", encoding="utf-8") as f:
    exec(f.read())

# README 파일 읽기
try:
    with open("README_REFACTORED.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = __description__

setup(
    name="retrieval",
    version=__version__,
    author=__author__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-google-genai",
        "langchain-chroma",
        "langchain-text-splitters",
        "langchain-core",
        "chromadb",
        "python-dotenv",
        "docx2txt",
        "sentence-transformers",
        "torch",
        "transformers"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
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
    keywords="rag retrieval-augmented-generation llm langchain chroma",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/retrieval/issues",
        "Source": "https://github.com/yourusername/retrieval",
        "Documentation": "https://github.com/yourusername/retrieval#readme",
    },
    include_package_data=True,
    zip_safe=False,
) 