import numpy as np
import os
import json
import torch
import faiss
import torchvision.models as models
from torch.autograd import Variable
from pprint import pprint
import torchvision.transforms as transforms

from PIL import Image


model = models.resnet50(pretrained=True)

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


# Transforms images to feature vectors

def image_to_vector(image_path):

    image = Image.open(image_path)

    t_image = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0))

    layer = model.fc

    model.eval()

    my_embedding = torch.zeros(1, 1000)

    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())

    h = layer.register_forward_hook(copy_data)

    model(t_image)
    h.remove()

    return my_embedding


# Create Image data base for search

def db_creation(dimension, db_size, image_db_path):

    index = faiss.index_factory(dimension, 'IDMap,Flat')
    paths_dictionary = {}
    for i, filename in enumerate(os.listdir(image_db_path), start=0):

        if i > db_size:
            break

        feature_vector = np.array(image_to_vector(f'{image_db_path}/{filename}'))

        image_id = np.array([i], dtype=np.int64)

        index.add_with_ids(feature_vector, image_id)

        paths_dictionary.update({int(image_id): f'{image_db_path}/{filename}'})

        pprint(paths_dictionary)

    with open('paths.json', 'w')as f:
        json.dump(paths_dictionary, f)

    return index
