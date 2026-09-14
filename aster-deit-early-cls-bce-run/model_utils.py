# /aster-deit/model_utils.py

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
import timm
from timm.data import create_transform

import config


def load_model_and_processor():
    """
    Loads the DeiT model and its corresponding image processor/transform.
    The transform is created using `timm` to match the original DeiT training recipe.
    """
    print(f"--- Loading Model and Processor for {config.MODEL_ID} ---")

    device = config.DEVICE

    # 1. Load Model from local path
    print(f"Loading model from local path: {config.MODEL_ID}")
    model = AutoModelForImageClassification.from_pretrained(
        config.MODEL_ID,
        output_hidden_states=True  # Crucial for ASTER
    ).to(device)
    model.eval()  # Base model is frozen and used for feature extraction

    print(f"Creating data transforms using 'timm' recipe for '{config.TIMM_MODEL_NAME}'")
    timm_model_for_cfg = timm.create_model(config.TIMM_MODEL_NAME, pretrained=False)
    timm_config = timm_model_for_cfg.default_cfg

    transform = create_transform(
        input_size=timm_config['input_size'],
        is_training=False,  # We are doing inference-style training of adapters
        interpolation=timm_config['interpolation'],
        mean=timm_config['mean'],
        std=timm_config['std'],
        crop_pct=timm_config['crop_pct'],
    )
    print("Model and image transform loaded successfully.")


    return model, transform


def get_backbone(model):
    if hasattr(model, "deit"):
        return model.deit
    if hasattr(model, "vit"):
        return model.vit
    raise AttributeError("Expected a Hugging Face DeiT/ViT model with .deit or .vit backbone")


def get_encoder_layers(model):
    backbone = get_backbone(model)
    candidates = [
        ("encoder.layer", lambda: backbone.encoder.layer),
        ("encoder.layers", lambda: backbone.encoder.layers),
        ("encoder.blocks", lambda: backbone.encoder.blocks),
        ("layers", lambda: backbone.layers),
        ("blocks", lambda: backbone.blocks),
    ]
    for _, getter in candidates:
        try:
            layers = getter()
            if isinstance(layers, (nn.ModuleList, list, tuple)):
                return layers
        except AttributeError:
            pass

    for name, module in backbone.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first = module[0]
            if hasattr(first, "attention") or hasattr(first, "layernorm_before"):
                print(f"Using encoder layers from backbone module list: {name}")
                return module

    children = [name for name, _ in backbone.named_children()]
    raise AttributeError(
        "Could not find DeiT/ViT encoder layers. "
        f"Backbone type={type(backbone).__name__}, children={children}"
    )


def run_embeddings(model, images):
    backbone = get_backbone(model)
    if hasattr(backbone, "embeddings"):
        return backbone.embeddings(images)
    if hasattr(backbone, "patch_embed"):
        return backbone.patch_embed(images)
    raise AttributeError(f"Could not find image embeddings module on {type(backbone).__name__}")
