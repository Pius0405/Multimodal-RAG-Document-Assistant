import fitz
from PIL import Image
import io


class PDFProcessor:
    def __init__(self, pdf_file, captioner):
        self.doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        self.captioner=captioner

    def extract_text(self):
        text_chunks = []
        for page_num, page in enumerate(self.doc):
            text = page.get_text()
            if text.strip():
                text_chunks.append({
                    "type": "text",
                    "page": page_num + 1,
                    "content": text
                })
        return text_chunks
    
    def extract_images_as_text(self):
        image_chunks = []
        all_images = []
        metadata = []

        # Collect all images first
        for page_num, page in enumerate(self.doc):
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                image_bytes = self.doc.extract_image(xref)["image"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # ensure RGB
                all_images.append(image)
                metadata.append({"page": page_num + 1, "index": img_index})

        if not all_images:
            return []

        # Batch caption all images
        descriptions = self.captioner.describe_batch(all_images, max_new_tokens=50)

        # Combine descriptions with metadata
        for desc, meta in zip(descriptions, metadata):
            image_chunks.append({
                "type": "image",
                "page": meta["page"],
                "content": desc
            })

        return image_chunks

    def process(self):
        # Combine text + image descriptions
        return self.extract_text() + self.extract_images_as_text()
