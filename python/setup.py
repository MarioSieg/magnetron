# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

# To install locally, use: pip3 install . --force-reinstall

import os
import subprocess
import multiprocessing

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

BUILD_RELEASE: bool = True
CMAKE_ROOT: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)  # Root directory of the CMake project
NUM_JOBS: int = max(multiprocessing.cpu_count() - 1, 1)  # Use all but one core


class BuildException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class CMakeBuildExtension(Extension):
    def __init__(self, name, root_dir: str = ''):
        super().__init__(name, sources=[])
        self.root_dir = os.path.abspath(root_dir)


class CMakeBuildExecutor(build_ext):
    def initialize_options(self):
        super().initialize_options()

    def run(self):
        try:
            print(subprocess.check_output(['cmake', '--version']))
        except OSError:
            raise BuildException(
                'CMake must be installed to build the magnetron binaries from source. Please install CMake and try again.'
            )
        super().run()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cmake_args = [
            '-DMAGNETRON_ENABLE_CUDA=OFF',  # TODO: Fix cuda compilation
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(os.path.join(self.build_lib, "magnetron"))}',
            f'-DCMAKE_BUILD_TYPE={"Release" if BUILD_RELEASE else "Debug"}',
        ]
        build_args = [
            '--target magnetron',  # Only build the magnetron library
            f'-j{NUM_JOBS}',
            '-v',
        ]
        print(
            subprocess.check_call(
                ['cmake', ext.root_dir] + cmake_args, cwd=self.build_temp
            )
        )
        print(
            subprocess.check_call(
                ['cmake', '--build', '.'] + build_args, cwd=self.build_temp
            )
        )


# Setup dependencies from requirements.txt
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f'{lib_folder}/requirements.txt'
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# Setup magnetron package
setup(
    name='magnetron',
    version='0.1.0',
    author='Mario Sieg',
    author_email='mario.sieg.64@gmail.com',
    description='A lightweight machine learning library with GPU support.',
    long_description='A lightweight machine learning library with GPU support.',
    packages=['magnetron'],
    package_dir={'': 'src'},
    package_data={
        'magnetron': ['*.dylib', '*.so', '*.dll'],
    },
    include_package_data=True,
    ext_modules=[CMakeBuildExtension('magnetron', root_dir=CMAKE_ROOT)],
    cmdclass={
        'build_ext': CMakeBuildExecutor,
    },
    zip_safe=False,
    install_requires=install_requires,
)
