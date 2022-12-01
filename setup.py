from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='simple_converge',
    version='0.11.0',
    description='utilities for faster and easier prototyping of DL models',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='MIT',
    packages=find_packages(),
    author='Eytan Kats',
    author_email='eytan.kats@gmail.com',
    keywords=['DeepLearning'],
    url='https://github.com/EytanKats/simple_converge',
    download_url='https://pypi.org/project/simple_converge/'
)

install_requires = [

]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
