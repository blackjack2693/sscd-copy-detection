import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sscd.models.model import Model
import lmdb
import faiss
import io

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])
torch.device("mps")
# Load the model
model = torch.jit.load("sscd_disc_advanced.torchscript.pt")
model.eval()  # Ensure model is in evaluation mode

embeddings = {}

# Open LMDB environment
with lmdb.open("/Users/Johann/masterproject/data/plankton/-TRAIN_imgs", readonly=True, lock=False) as env:
    with env.begin() as txn:  # Start a transaction
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            img = Image.open(io.BytesIO(value)).convert('RGB')  # Fix: Load from bytes
            batch = small_288(img).unsqueeze(0)

            with torch.no_grad():
                emb = model(batch)  # Extract embedding

            embeddings[key] = emb
            # print(emb)
            if i >= 100:
                break
print("Done")
print(len(embeddings))
# Convert embeddings to a tensor
# Convert embeddings to a tensor (without squeezing)
keys = list(embeddings.keys())
vectors = torch.stack([embeddings[k] for k in keys])  # Shape is (N, 1, 512)

# Compute cosine similarity over the last dimension (-1)
similarity_matrix = torch.nn.functional.cosine_similarity(
    vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1
)
# This gives a shape of (N, N, 1); you may want to squeeze the extra dimension:
similarity_matrix = similarity_matrix.squeeze(-1)  # Now shape is (N, N)

# Get the top k similar images for each image
k = 5
top_similarities, top_indices = torch.topk(similarity_matrix, k=k, dim=1)

# Print results, skipping self-similarity
for i, key in enumerate(keys):
    print(f"\nSimilar images to {key.decode()}:")
    for sim, idx in zip(top_similarities[i], top_indices[i]):
        if idx.item() == i:
            continue
        print(f"Image: {keys[idx.item()].decode()}, Similarity: {sim:.3f}")

# index = faiss.IndexFlatIP(dim)
# index.add(np.ascontiguousarray(normalized_np))

# # Thresholding setup
# threshold = 0.9
# results = {}

# # Query the index
# for i, key in enumerate(keys):
#     distances, indices = index.search(normalized_np[i].reshape(1, -1), 50)
#     print(distances)
#     similar = [
#         keys[idx] for dist, idx in zip(distances[0], indices[0]) if dist > threshold and idx != i
#     ]
#     results[key] = similar

# print(results)
