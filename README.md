# WearifAI | AI Fashion Cloth Changer

**WearifAI**, a ground-breaking technology that promises to solve long-standing issues in the clothing industry, particularly on addressing limited virtual try-on alternatives during the COVID-19 pandemic, and visibility limits for up-and-coming designers.
WearifAI transforms fashion design into a dynamic canvas by utilising generative AI, which benefits both consumers and creators. 

Important features include size matching, which uses AI-driven automation to tag clothing items and establish user size and gender, and smart body fit, which guarantees exact recognition of various body parts for accurate virtual depiction. When arranging garments virtually, realistic clothing placement uses cutting-edge technology to create a realistic and natural appearance.

Moreover, users may choose different clothing kinds, submit pictures, and let the AI do the image processing. Consumers have the freedom to try on virtual outfits whenever it's convenient for them.

Our AI training model goes through a streamlined process to generate advantageous results, this includes gathering datasets of product images, cleaning and analysing data (each item is tagged) , training models with Dreambooth (DreamLike Photoreal v2 inpainting model), making model inference with necessary inputs, and carefully creating images using particular apparel, with the ControlNet act as guidance. Furthermore, we also store the data in Google Drive for better analysis in the future.
We believe, WearifAI stands for high feasibility, which provides a virtual fitting revolution, insightful data, global accessibility, metaverse trend alignment, and different business needs, all in line with growing market trends.

## Test
```
from diffusers import DiffusionPipeline
import torch

model_id = "model/2880"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of a man wears cottonXcasualXhoodieXblue hoodie"
image = pipe(prompt, num_inference_steps=50, guidance_scale=9, width=512,height=512).images[0]

image.save("person-cottonXcasualXhoodieXblue-hoodie.png")
image

```
![image](https://github.com/user-attachments/assets/028f7d7e-1ca6-451f-a957-535107be18a4)


## Our model workflow
![image](https://github.com/user-attachments/assets/89adefbe-e896-483f-901b-673f16766814)
