# MVDream-hf

modified from https://github.com/KokeCacao/mvdream-hf.

### convert weights

MVDream:
```bash
# dependency
pip install -U omegaconf diffusers safetensors huggingface_hub transformers accelerate

# download original ckpt
cd models
wget https://huggingface.co/MVDream/MVDream/resolve/main/sd-v2.1-base-4view.pt
wget https://raw.githubusercontent.com/bytedance/MVDream/main/mvdream/configs/sd-v2-base.yaml
cd ..

# convert
python convert_mvdream_to_diffusers.py --checkpoint_path models/sd-v2.1-base-4view.pt --dump_path ./weights_mvdream --original_config_file models/sd-v2-base.yaml --half --to_safetensors --test
```

ImageDream:
```bash
# download original ckpt
wget https://huggingface.co/Peng-Wang/ImageDream/resolve/main/sd-v2.1-base-4view-ipmv-local.pt
wget https://raw.githubusercontent.com/bytedance/ImageDream/main/extern/ImageDream/imagedream/configs/sd_v2_base_ipmv_local.yaml

# convert
python convert_mvdream_to_diffusers.py --checkpoint_path models/sd-v2.1-base-4view-ipmv-local.pt --dump_path ./weights_imagedream --original_config_file models/sd_v2_base_ipmv_local.yaml --half --to_safetensors --test
```

### usage

example:
```bash
python run_mvdream.py "a cute owl"
python run_imagedream.py data/anya_rgba.png
```