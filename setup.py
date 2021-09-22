from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='simple_converge',
    version='0.7.0',
    description='framework for configurable training DL models',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='MIT',
    packages=find_packages(),
    author='Eytan Kats',
    author_email='eytan.kats@gmail.com',
    keywords=['DeepLearning'],
    url='https://github.com/EytanKats/dl_framework',
    download_url='https://pypi.org/project/simple_converge/'
)

install_requires = [
    'pandas>=1.2.2',
    'opencv-python>=4.4.0.46',
    'scikit-image>=0.17.2',
    'matplotlib>=3.3.2',
    'seaborn>=0.11.0',
    'tensorflow==2.6.0',
    'clearml>=1.0.5',
    'segmentation-models==1.0.1',
    'classification-models==0.1',
    'efficientnet==1.0.0',
    'keras-unet-collection==0.0.18'
    # 'image-classifiers==1.0.0b1'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)