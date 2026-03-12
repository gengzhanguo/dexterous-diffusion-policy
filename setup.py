from setuptools import setup, find_packages

setup(
    name="dexterous_diffusion_policy",
    version="0.1.0",
    description="Diffusion Policy for Dexterous Manipulation on a Single GPU",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "gymnasium-robotics>=1.2.4",
        "mujoco>=3.0.0",
        "h5py>=3.9.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
        "omegaconf>=2.3.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        "einops>=0.7.0",
        "rich>=13.5.0",
    ],
)
