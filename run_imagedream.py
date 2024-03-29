import torch
import kiui
import numpy as np
import argparse
from pipeline_mvdream import MVDreamPipeline

pipe = MVDreamPipeline.from_pretrained(
    # "./weights_imagedream", # local weights
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
pipe = pipe.to("cuda")


parser = argparse.ArgumentParser(description="ImageDream")
parser.add_argument("image", type=str, default='data/anya_rgba.png')
parser.add_argument("--prompt", type=str, default="")
args = parser.parse_args()

for i in range(5):
    input_image = kiui.read_image(args.image, mode='float')
    image = pipe(args.prompt, input_image, guidance_scale=5, num_inference_steps=30, elevation=0)
    grid = np.concatenate(
        [
            np.concatenate([image[0], image[2]], axis=0),
            np.concatenate([image[1], image[3]], axis=0),
        ],
        axis=1,
    )
    # kiui.vis.plot_image(grid)
    kiui.write_image(f'test_imagedream_{i}.jpg', grid)
