import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='jupyters3',
    version='0.0.20',
    author='Department for International Trade - WebOps',
    author_email='webops@digital.trade.gov.uk',
    description='Jupyter Notebook Contents Manager for AWS S3',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/uktrade/jupyters3',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
