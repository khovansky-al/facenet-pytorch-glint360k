import torch
import torchvision.transforms as transforms
from torch.multiprocessing import Process, Queue, set_start_method
import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path
import os
import math

from models.resnet import Resnet34Triplet
from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import MTCNN
from PIL import Image

device = torch.device("cpu")
# device = torch.device("mps")

checkpoint = torch.load(
    './model/model_resnet34_triplet.pt',
    map_location=device,
    weights_only=False,
)
model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
model.load_state_dict(checkpoint['model_state_dict'])
best_distance_threshold = checkpoint['best_distance_threshold']

model.to(device)
model.eval()

preprocess = transforms.Compose([
  # transforms.ToPILImage(),
  #!!
  # transforms.Resize(size=140),  # Pre-trained model uses 140x140 input images
  # transforms.ToTensor(),
  transforms.Normalize(
      mean=[0.6071, 0.4609, 0.3944],  # Normalization settings for the model, the calculated mean and std values
      std=[0.2457, 0.2175, 0.2129]     # for the RGB channels of the tightly-cropped glint360k face dataset
  )
])

# mtcnn = MTCNN(model_dir='model', margin=0, image_size=224)

#!!
mtcnn = MTCNN(model_dir='model', margin=0, image_size=140, device=device)
# mtcnn = MTCNN(model_dir='model', margin=0, image_size=140)
mtcnn.to(device)
mtcnn.eval()

# old_model = InceptionResnetV1('model').eval()
# old_model.to(device)
# old_model.eval()

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def get_embedding(image_path):
    with torch.no_grad():
        img = Image.open(image_path).convert('RGB')
        img = mtcnn(img)
        # img = mtcnn(img, save_path=f"{image_path}_p.jpeg")

        # TODO: only for the new model?
        img = preprocess(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        embedding = model(img)

        # Turn embedding Torch Tensor to Numpy array
        return embedding.cpu().detach().numpy()

def l2_distance(emb1, emb2):
    return np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))

# max_filename_length = max(len(file.name) for file in image_files)
# col_width = max(max_filename_length + 2, 10)

def generate_pairs(image_files):
    pairs = []

    for i in range(n):
        file1 = image_files[i]

        for j in range(i + 1, n):
            file2 = image_files[j]

            pairs.append((file1, file2))

    return pairs

def get_distance(file1, file2):
    # print(file1, file2)
    embedding1 = get_embedding(file1)
    embedding2 = get_embedding(file2)

    return (l2_distance(embedding1, embedding2), file1, file2)

def get_distance_map(q, tasks):
    for task in tasks:
        q.put(get_distance(*task))
        # if (diff < threshold):
        #     positives+=1

        # if (diff < 0.5):
        #     outliers.append((file1.name, file2.name, diff))

    # # Print column headers
    # print(" " * (col_width), end="")
    # for i in range(n):
    #     print(f"{image_files[i].name:>{col_width}}", end="")
    # print()

    # # Print matrix with row labels
    # for i in range(n):
    #     print(f"{image_files[i].name:<{col_width}}", end="")
    #     for j in range(n):
    #         print(f"{distance_matrix[i,j]:>{col_width}.4f}", end="")
    #     print()

if __name__ == '__main__':
    folder = sys.argv[1]
    image_files = []
    valid_extensions = {'.jpg', '.jpeg'}

    for file in Path(folder).iterdir():
        if file.suffix.lower() in valid_extensions:
            image_files.append(file)

    image_files.sort()
    n = len(image_files)
    total_pairs = n * (n - 1) / 2

    pairs = generate_pairs(image_files)

    positives = 0
    outliers = []

    num_cpus = os.cpu_count() or 1
    q = Queue()

    single_chunk = math.ceil(total_pairs / num_cpus)

    processes = []
    distances = []

    for i in range(num_cpus):
        start = i * single_chunk
        chunk = pairs[start:start + single_chunk]
        print(f"{i}, {len(chunk)}")

        p = Process(target=get_distance_map, args=(q, chunk))
        processes.append(p)
        p.start()

    processed = 0

    with tqdm(total=total_pairs) as t:
        while True:
            distances.append(q.get())
            processed += 1
            t.update()

            if processed == total_pairs:
                break


    for p in processes:
        p.join()

    for distance in distances:
        diff, file1, file2 = distance

        if diff < 0.8:
            positives += 1
        if diff < 0.5:
            outliers.append((file1, file2, distance))

    percent_false = (positives / total_pairs) * 100
    print(f"Positives: {positives}/{total_pairs}, {percent_false:.2f}%")
    print(f"Outliers: {outliers}")


    # if (sys.argv[2] == 'old'):
    #     print('FaceNet VGGFace')
    #     print("----------------")
    #     process_and_print(old_model, 0.8)
    # else:
    #     print('FaceNet Glint360K')
    #     print("----------------")
    #     process_and_print(model, 0.8)
