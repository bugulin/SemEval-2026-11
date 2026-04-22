"""
Training of custom models.
"""

import torch


def check_gpu() -> None:
    print(f"Working on GPU {torch.cuda.get_device_name()}")
