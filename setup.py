from setuptools import setup, find_packages

setup(
    name="simple_gelsight",
    version = "0.1",
    packages = find_packages(),
    install_requires = ["opencv-python",
                        "numpy"]
)
