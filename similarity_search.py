import numpy as np
import faiss
import db_creation

# Perform index search

def index_search(target_image_path, k):
    index = faiss.read_index('faiss.index')
    target_image = np.array(db_creation.image_to_vector(target_image_path))
    D, I = index.search(target_image, k)

    return I
