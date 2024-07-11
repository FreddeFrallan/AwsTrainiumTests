# How to fetch and tokenize the data

## Fetch the data

We are using this dataset: https://huggingface.co/datasets/bigcode/the-stack/tree/main/data/c

From here, we only need one file (I used train-00000-of-00257.parquet)


It contains ~360MB of C code. Fetch it and store locally.

## Tokenize the data

Use __tokenize_data.py__ script. Example:

```bash
python3 tokenize_data.py --data_file train-00000-of-00257.parquet --tokenization_folder tokenized_data
```

The script takes more args, but most of them should remain default (it is a super-simple script so feel free to take a look and modify according to your needs).

An important detail is that on my local machine I could not fit the entire raw code + tokenized code in memory, so I chunked the input into smaller parts. You can use __chunk_size__ parameter to control it - if the deafult params do not work, try to decrease the chunk size.

When using 'only' 100M characters from the raw code, we end up with ~40M tokens which should give us reasonable number of samples (~27K for seq_length=3*512) for our tests. To control how many characters to include, use __chars_to_tokenize__ which is 100M by default.


## Data format and usage

In the end, in the __tokenization_folder__ you will see dump-{idx}.pkl pickled files and a dump-meta_data.json file.
The latter tells you how many files, samples and tokens you have in the dataset. The dump-{idx}.pkl files is a pickled list of samples (also lists) of lengths __seq_length__ (3*512 by default).

This data formt is directly consumable by PickledTrainerDataset from pickle_dataset. You can load the data (see the training scripts) using

```python
In [1]: from pickle_dataset import PickledTrainerDataset

In [2]: ptd = PickledTrainerDataset(["testme"])
Finding files:   0%|                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]Checking folder: testme
Found meta file: testme/dump-meta_data.json
Finding files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2286.97it/s]
Checked 1 folders and found 10 files with extension: pkl
Total samples: 27290
Total tokens: 41917440
Loading tokenized data: 0it [00:00, ?it/s]Loaded 2759 vectors from testme/dump-0.pkl
Loaded 2691 vectors from testme/dump-4.pkl
Loading tokenized data: 2it [00:00,  9.39it/s]Loaded 2922 vectors from testme/dump-1.pkl
Loaded 2631 vectors from testme/dump-3.pkl
Loading tokenized data: 4it [00:00, 10.85it/s]Loaded 2658 vectors from testme/dump-6.pkl
Loaded 2536 vectors from testme/dump-8.pkl
Loading tokenized data: 6it [00:00, 10.01it/s]Loaded 2587 vectors from testme/dump-9.pkl
Loaded 2511 vectors from testme/dump-5.pkl
Loading tokenized data: 8it [00:00, 10.83it/s]Loaded 3063 vectors from testme/dump-2.pkl
Loaded 2932 vectors from testme/dump-7.pkl
Loading tokenized data: 10it [00:00, 10.12it/s]

In [3]: len(ptd)
Out[3]: 27290

In [4]: ptd[0]["input_ids"].shape
Out[4]: torch.Size([1536])
```

## Validation data

In order to generate validation data, use **starting_character** switch to start from some index beyond the **hars_to_tokenize** scope of the trainig data. Example:

`python3 tokenize_data.py --data_file train-00000-of-00257.parquet --tokenization_folder validation_512 --seq_length 512 --starting_character 100000000 --chars_to_tokenize 2000000`
 