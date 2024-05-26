import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="champion",
    version="1.1.0",
    author="Pratanu",
    author_email="mitra.pratanu@gmail.com",
    description="Portfolio Evaluation with ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages()
)