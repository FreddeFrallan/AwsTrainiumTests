from torch.utils.data import Dataset
from torch.nn import functional as F
import tqdm
import os
import json
from torch import tensor
import pickle

class PickledTrainerDataset(Dataset):
    def __init__(self, target_folders):
        self.target_folders = target_folders
        self.files = self._locate_dataset()
        self._load_dataset()

    def _locate_dataset(self):
        files = []

        for folder in tqdm.tqdm(self.target_folders, desc='Finding files'):
            print('Checking folder:', folder)
            folder_meta_file = None
            folder_data_files = {}

            # Find all related files
            for file in os.listdir(folder):
                if file.endswith('pkl'):
                    folder_data_files[file] = os.path.join(folder, file)
                if file.endswith('.json'):
                    print('Found meta file:', os.path.join(folder, file))
                    with open(os.path.join(folder, file), 'r') as meta_file:
                        folder_meta_file = json.load(meta_file)

            if folder_meta_file is None:
                raise ValueError('No meta file found for folder:', folder)

            for f in folder_data_files:
                assert f in folder_meta_file, f'Missing meta data for file: {f}'
                files.append((folder_data_files[f], folder_meta_file[f]))              

        # Print statistics about the collected dataset
        print(
            f"Checked {len(self.target_folders)} folders and found {len(files)} files with extension: {'pkl'}")
        self.total_samples = sum([meta['num_samples'] for _, meta in files])
        print(f"Total samples: {self.total_samples}")
        # self.total_batches = self.total_samples // self.batch_size
        # print(f"Total batches: {self.total_batches}")
        self.total_tokens = sum([meta['num_tokens'] for _, meta in files])
        print(f"Total tokens: {self.total_tokens}")

        return files

    def _load_dataset(self):
        self.data = []
        for f in tqdm.tqdm(map(lambda x: x[0], self.files), desc='Loading tokenized data'):
            with open(f, 'rb') as file:
                tokenized_data = pickle.load(file)
                print(f"Loaded {len(tokenized_data)} vectors from {f}")
                self.data.extend(tokenized_data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # return {
        #     "input_ids": F.pad(tensor(sample)[...,:-1], (0, 1), mode='constant', value=32018),
        #     "labels": F.pad(tensor(sample)[...,1:], (0, 1), mode='constant', value=32018)
        # }
        return {
            "input_ids": tensor(sample),
            "labels": tensor(sample) # same as inputs, will be shifted in the masked_loss
        }