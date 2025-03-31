# Qwen Vision Language Model (VLM) Tutorial

This repository contains materials for learning and experimenting with the Qwen2.5-VL model, a powerful vision-language model that can process both text and images.

## Directory Structure

```
qwen-vl/
├── .venv/                  # Virtual environment directory
├── qwen_vl_tutorial.py     # Main tutorial script for Qwen VLM
├── qwen_vl_tutorial.ipynb  # Jupyter notebook version of the tutorial
├── qwen_vl_mlx.ipynb       # MLX-specific notebook for Qwen VLM
├── qwen-inference.py       # Reference script for Qwen VLM inference
├── pyproject.toml          # Project configuration for uv
├── requirements.txt        # Legacy package requirements
└── uv.lock                 # Lock file for uv dependencies
```

## Getting Started

### Prerequisites

1. Install uv - the fast Python package installer and virtual environment manager:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   
   Or with pip:
   ```bash
   pip install uv
   ```

### Setup

1. Create a virtual environment using uv:
   ```bash
   uv venv
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install all required dependencies using uv sync:
   ```bash
   uv sync
   ```
   
   This will install all dependencies specified in the pyproject.toml file, including:
   - torch and torchvision
   - transformers
   - matplotlib
   - ipykernel and ipywidgets
   - mlx-vlm (from GitHub repository)

### Running the Tutorial

1. Run the tutorial script:
   ```bash
   python qwen_vl_tutorial.py
   ```

2. For interactive exploration, start Jupyter notebook:
   ```bash
   jupyter notebook
   ```
   
   Then open either `qwen_vl_tutorial.ipynb` or `qwen_vl_mlx.ipynb`.

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
- [uv Documentation](https://github.com/astral-sh/uv)
