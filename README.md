# LLM Runner

## Installation

```
sudo singularity build llm-pytorch-cuda11.3.sif singularity.def
```

## Quick Start

```
singularity exec --nv llm-pytorch-cuda11.3.sif sh -c python run_llm.py
```