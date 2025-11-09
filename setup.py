"""
Setup script for Explainable Sentiment Analysis package.
"""

from setuptools import setup, find_packages

setup(
    name="explainable-sentiment-analysis",
    version="1.0.0",
    description="Explainable Sentiment Analysis with Integrated Gradients, SHAP, and Counterfactuals",
    author="XSA Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "shap>=0.42.0",
        "captum>=0.6.0",
        "nltk>=3.8.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)

