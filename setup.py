from setuptools import setup, find_packages

setup(
    name="aerial-unet-segmentation",
    version="1.0.0",
    description="U-Net deep learning model for aerial image segmentation",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/YOUR_USERNAME/aerial-unet-segmentation",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "albumentations>=1.3.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "tensorboard>=2.12.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    entry_points={
        "console_scripts": [
            "unet-train=src.train:main",
            "unet-predict=src.predict:main",
            "unet-evaluate=src.evaluate:main",
        ]
    },
)
