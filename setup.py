import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class InSourceCMakeExtension(Extension):
    """A dummy Extension that tells setuptools we are using CMake."""
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class InSourceCMakeBuild(build_ext):
    """Custom build_ext that runs CMake in‑source."""
    def run(self):
        # Verify CMake is installed
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError("CMake must be installed to build the extensions") from exc

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: InSourceCMakeExtension):
        # Destination where the compiled .so must end up‑front
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        cfg = "Debug" if self.debug else "Release"

        # -----------------------------------------------------------------
        # 1️⃣  Ensure a clean in‑source build directory
        # -----------------------------------------------------------------
        src_dir = Path(ext.sourcedir)

        # Remove stale CMake artefacts that could confuse a fresh configure
        for stale in ("CMakeCache.txt", "CMakeFiles", "Makefile", "cmake_install.cmake"):
            path = src_dir / stale
            if path.is_dir():
                subprocess.check_call(["rm", "-rf", str(path)])
            elif path.is_file():
                path.unlink(missing_ok=True)

        # -----------------------------------------------------------------
        # 2️⃣  CMake configure step (source == binary)
        # -----------------------------------------------------------------
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        # -S <src> -B <src> tells CMake that the build dir is the same as the source dir
        subprocess.check_call(
            ["cmake", "-S", str(src_dir), "-B", str(src_dir), "-G", "Ninja"] + cmake_args,
            cwd=src_dir,
            )

        # -----------------------------------------------------------------
        # 3️⃣  Build the target
        # -----------------------------------------------------------------
        subprocess.check_call(
            ["cmake", "--build", str(src_dir), "--config", cfg, "--target", ext.name, "--parallel", str(os.cpu_count())],
            cwd=src_dir,
        )

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = os.path.join(lib_folder, 'requirements.txt')
install_requires = []

if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name="pyquicksr",
    version="0.1.0",
    author="Doğu Kocatepe",
    description="Python wrapper for QuickSR",
    packages=["quicksr"],
    package_dir={"": "python"},
    ext_modules=[InSourceCMakeExtension("quicksr", sourcedir=".")],
    cmdclass={"build_ext": InSourceCMakeBuild},
    zip_safe=False,
    install_requires=install_requires
)
