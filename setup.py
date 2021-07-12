from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='simple_converge',
    version='0.6.1',
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
    'tensorflow==2.3.0',
    'segmentation-models==1.0.1',
    'classification-models==0.1',
    'keras-unet-collection==0.0.18'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)