from setuptools import setup, find_packages

setup(
    name='FinanceSentimentAnalyzer',  # Replace with your project name
    version='0.0.1',  # Version of your project
    packages=find_packages(),
    include_package_data=True,
    description='Predict positive or negative sentiment from financial news headlines.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Matthew Wong',
    author_email='wongmatthew6767@gmail.com',
    url='https://github.com/MatthewW05/FinanceSentimentAnalyzer',
    install_requires=[
        'torch<2.4',
        'torchtext==0.18.0',
        'pandas==2.2.2',
        'numpy<=1.26.4',
        'scikit-learn==1.5.1',
        'chardet==5.2.0',
    ],
    python_requires='>=3.8,<3.12',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='machine learning, data processing, PyTorch, sentiment-analysis, financial-news',
)
