import torch
import kiui
import numpy as np
import argparse
from mvdream.pipeline_mvdream import MVDreamPipeline

pipe = MVDreamPipeline.from_pretrained(
    "./weights_imagedream", # local weights
    # "ashawkey/mvdream-sd2.1-diffusers",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")


parser = argparse.ArgumentParser(description="ImageDream")
parser.add_argument("image", type=str, default='data/anya_rgba.png')
parser.add_argument("--prompt", type=str, default="")
args = parser.parse_args()

for i in range(5):
    input_image = kiui.read_image(args.image, mode='float')
    image = pipe(args.prompt, input_image, guidance_scale=5)
    grid = np.concatenate(
        [
            np.concatenate([image[0], image[2]], axis=0),
            np.concatenate([image[1], image[3]], axis=0),
        ],
        axis=1,
    )
    # kiui.vis.plot_image(grid)
    kiui.write_image(f'test_imagedream_{i}.jpg', grid)
