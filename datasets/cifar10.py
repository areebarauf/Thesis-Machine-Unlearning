import tarfile
import torchvision
from torchvision.datasets.utils import download_url


# Dowload the dataset
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
download_url(dataset_url, '.')

# Extract from archive
with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    tar.extractall(path='/media/homes/areeba/thesis_projects/FYEMU/datasets/')

# Look into the data directory
data_dir = '/media/homes/areeba/thesis_projects/FYEMU/datasets'

