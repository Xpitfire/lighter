from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    readme = fh.read()

requirements = {"install": ["ffmpeg", "natsort", "numpy", "tqdm", "imageio", "opencv-python",
                            "torch", "torchvision", "facenet-pytorch", "tensorboard", "coloredlogs",
                            "petname", "pandas"]}

install_requires = requirements["install"]

setup(
    # Metadata
    name="lighter",
    author="Marius-Constantin Dinu",
    version="0.0.1",
    author_email="dinu.marius-constantin@hotmail.com",
    url="https://github.com/Xpitfire/lighter",
    description="Lightweight reinforcement learning framework for PyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    # Package info
    packages=find_packages(),
    zip_safe=True,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent"
    ]
)
