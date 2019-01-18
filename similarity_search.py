import numpy as np
import torch
import faiss
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

from PIL import Image


model = models.resnet50(pretrained=True)

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


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


# Perform index search

# print(I[-5:])
def index_search(target_image_path, k):
    index = faiss.read_index('faiss.index')
    target_image = np.array(image_to_vector(target_image_path))
    D, I = index.search(target_image, k)

    print(I[:5])
    return I
