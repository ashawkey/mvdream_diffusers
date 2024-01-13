import torch
import kiui
import numpy as np
import argparse
from mvdream.pipeline_mvdream import MVDreamPipeline

pipe = MVDreamPipeline.from_pretrained(
    "./weights", # local weights
    # "ashawkey/mvdream-sd2.1-diffusers",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")


parser = argparse.ArgumentParser(description="MVDream")
parser.add_argument("prompt", type=str, default="a cute owl 3d model")
args = parser.parse_args()

while True:
    image = pipe(args.prompt)
    grid = np.concatenate(
        [
            np.concatenate([image[0], image[2]], axis=0),
            np.concatenate([image[1], image[3]], axis=0),
        ],
        axis=1,
    )
    kiui.vis.plot_image(grid)
