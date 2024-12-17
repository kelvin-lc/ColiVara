import torch
from PIL import Image

from colpali_engine.models import ColQwen2, ColQwen2Processor

base_model_name = (
    "/Users/lc/Documents/ai_models/hf_models/models--vidore--colqwen2-base"
)
model_name = "/Users/lc/Documents/ai_models/hf_models/models--vidore--colqwen2-v1.0"

model = ColQwen2.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="mps",  # or "mps" if on Apple Silicon
).eval()

model.load_state_dict(model_name)

processor = ColQwen2Processor.from_pretrained(base_model_name)

# Your inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
queries = [
    "Is attention really all you need?",
    "Are Benjamin, Antoine, Merve, and Jo best friends?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
