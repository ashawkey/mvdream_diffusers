# MVDream-hf

modified from https://github.com/KokeCacao/mvdream-hf.

### convert weights
```bash
# dependency
pip install -U omegaconf diffusers safetensors huggingface_hub transformers accelerate

# download original ckpt
wget https://huggingface.co/MVDream/MVDream/resolve/main/sd-v2.1-base-4view.pt
wget https://raw.githubusercontent.com/bytedance/MVDream/main/mvdream/configs/sd-v2-base.yaml

# convert
python convert_mvdream_to_diffusers.py --checkpoint_path ./sd-v2.1-base-4view.pt --dump_path ./weights --original_config_file ./sd-v2-base.yaml --half --to_safetensors --test
```

### usage

example:
```bash
python main.py "a cute owl"
```

detailed usage:
```python
import torch
import kiui
from mvdream.pipeline_mvdream import MVDreamPipeline

pipe = MVDreamPipeline.from_pretrained('./weights', torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt) # np.ndarray [4, 256, 256, 3]

kiui.vis.plot_image(image)
```