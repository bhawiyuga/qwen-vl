# Qwen Vision Language Model (VLM) Tutorial

This repository contains materials for learning and experimenting with the Qwen2.5-VL model, a powerful vision-language model that can process both text and images.

## Setup

The environment is set up using `uv`, a fast Python package installer and virtual environment manager.

1. The virtual environment has already been created in `.venv` directory
2. Required packages have been installed:
   - torch
   - transformers
   - jupyter
   - pillow
   - accelerate
   - qwen-vl-utils (official package)

## Files

- `qwen_vl_tutorial.py`: A comprehensive Python script that guides you through using the Qwen2.5-VL model
- `qwen-inference.py`: Reference script for Qwen VLM inference
- `requirements.txt`: List of dependencies for the project

## Getting Started

1. Activate the virtual environment:
   ```
   source .venv/bin/activate
   ```

2. Run the tutorial script:
   ```
   python qwen_vl_tutorial.py
   ```

3. For interactive exploration, start Jupyter notebook:
   ```
   jupyter notebook
   ```

## Tutorial Contents

The tutorial covers:

1. Environment setup
2. Loading the Qwen2.5-VL model
3. Image description
4. Visual question answering
5. Using local images
6. Multi-turn conversations
7. Customizing generation parameters

## Model Information

The tutorial uses the `Qwen/Qwen2.5-VL-3B-Instruct` model, which is a 3B parameter vision-language model fine-tuned for instruction following. The model can:

- Describe images in detail
- Answer questions about image content
- Compare multiple images
- Engage in multi-turn conversations about visual content
- Generate creative content based on images

## Resources

- [Qwen2.5 GitHub Repository](https://github.com/QwenLM/Qwen2.5)
- [Hugging Face Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
