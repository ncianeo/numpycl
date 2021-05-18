from setuptools import setup, find_packages

exec (open('npcl/version.py').read())

setup(
    name='npcl',
    version=__version__,
    description='Numerical Library Written in Python with PyOpenCL',
    url='https://github.com/ncianeo/numpycl',
    author='Seungwon Jeong',
    author_email='jsw1295@snu.ac.kr',
    license='MIT License',
    packages=find_packages(),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='image processing',

    install_requires=[
        'numpy>=1.11.0',
        'pyopencl>=2019.1',
        'mako',
        ],

    package_data={
        'npcl': [
            'ops/*.cl',
            'regularizers/*.cl',
            'solvers/*.cl',
            ],
        },

    entry_points={},
)
