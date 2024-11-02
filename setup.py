from setuptools import setup, find_packages

setup(
    name="roleplay-roleplay",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "python-dotenv>=0.19.0",
        "pydantic>=2.0.0",
    ],
    author="Comedian1926",
    description="A flexible framework for creating multi-roleplay roleplaying chat environments",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RolePlay-Agent",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)