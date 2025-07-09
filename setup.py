from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fitness-multi-agent",
    version="1.0.0",
    author="Enterprise AI Team",
    author_email="ai-team@company.com",
    description="企业级多智能体运动健身伴侣系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/company/fitness-multi-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Health :: Fitness",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "gpu": [
            "torch>=2.1.0+cu118",
            "torchvision>=0.16.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "fitness-agent=src.main:main",
            "fitness-api=src.api.main:start_server",
            "fitness-worker=src.workers.main:start_worker",
        ],
    },
)