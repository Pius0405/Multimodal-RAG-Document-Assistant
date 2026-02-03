from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

class ImageCaptionerHF:
    def __init__(self, model_name="nlpconnect/vit-gpt2-image-captioning"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def describe_batch(self, images, max_new_tokens=50):
        """
        Accepts a list of PIL images and returns a list of text descriptions
        """
        # Preprocess all images as a batch
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(pixel_values, max_new_tokens=max_new_tokens)

        # Decode each image's generated output
        descriptions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        return descriptions