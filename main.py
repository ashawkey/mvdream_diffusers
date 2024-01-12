import torch
import kiui
from mvdream.pipeline_mvdream import MVDreamStableDiffusionPipeline

pipe = MVDreamStableDiffusionPipeline.from_pretrained('./weights', torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt)

kiui.vis.plot_image(image)