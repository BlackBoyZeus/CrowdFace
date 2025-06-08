"""
BAGEL model loader for CrowdFace
"""

import os
import torch
from typing import Optional, Dict, Any, Tuple

def load_bagel_model(model_path: str, token: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Load the BAGEL model and inferencer
    
    Args:
        model_path: Path to BAGEL model directory
        token: Hugging Face token for accessing gated models
        
    Returns:
        Tuple of (model, inferencer)
    """
    try:
        from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
        from Bagel.data.transforms import ImageTransform
        from Bagel.data.data_utils import add_special_tokens
        from Bagel.modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
        )
        from Bagel.modeling.qwen2 import Qwen2Tokenizer
        from Bagel.modeling.bagel.qwen2_navit import NaiveCache
        from Bagel.modeling.autoencoder import load_ae
        from Bagel.inferencer import InterleaveInferencer
        
        # LLM config preparing
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        
        # ViT config preparing
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        
        # VAE loading
        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
        
        # Bagel config preparing
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config, 
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )
        
        # Initialize model with empty weights
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        
        # Load tokenizer and add special tokens
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        
        # Set up transforms
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)
        
        # Set up device map for model loading
        device_map = infer_auto_device_map(
            model,
            max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        
        # Define modules that should be on the same device
        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
        ]
        
        # Load model weights
        model = load_checkpoint_and_dispatch(
            model, 
            os.path.join(model_path, "pytorch_model.bin"),
            device_map=device_map,
            offload_folder=None,
            offload_state_dict=False,
            same_device_modules=same_device_modules,
        )
        
        # Initialize the inferencer
        bagel_inferencer = InterleaveInferencer(
            model=model, 
            vae_model=vae_model, 
            tokenizer=tokenizer, 
            vae_transform=vae_transform, 
            vit_transform=vit_transform, 
            new_token_ids=new_token_ids
        )
        
        print("BAGEL model loaded successfully!")
        return model, bagel_inferencer
        
    except Exception as e:
        print(f"Error loading BAGEL model: {e}")
        print("Will use fallback methods for scene understanding and ad placement.")
        return None, None
