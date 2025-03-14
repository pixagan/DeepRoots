from setuptools import setup, find_packages

setup(
    name="deeproots",  # Replace with your package name
    version="0.1.0",    # Initial version
    author="Anil Variyar",
    author_email="pixagan@gmail.com",
    description="Neural Network Builder and Visualizer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pixagan/deeproots",  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        "numpy", 
        "pandas",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version required
)
