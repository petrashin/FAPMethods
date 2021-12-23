from setuptools import find_packages, setup

setup(
    name='fapmethods',
    packages=find_packages(include=['fapmethods']),
    version='0.1.0',
    description='Very useful lib from FA University to all the world',
    author='Nikita, George, Maxim, Michael, Aydar',
    license='MIT',
    install_requires=[
        'numpy==1.21.5',
        'pandas==1.3.5',
        'PyWavelets==1.2.0',
        'scipy==1.7.3',
        'sympy==1.9'
    ]
)
