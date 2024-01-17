# LLM Runner

## Installation

```
sudo singularity build llm-pytorch-cuda12.1.sif singularity.def
```

## Quick Start

```
singularity exec --nv llm-pytorch-cuda12.1.sif sh -c python run_llm.py
```