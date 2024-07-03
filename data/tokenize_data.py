from transformers import AutoTokenizer
import pandas as pd
import pickle
import tqdm
import json
import argparse
import os


def chunk_and_tokenize(data, tokenizer, 
                       pad_token=32014, 
                       seq_length=512, 
                       min_sample_length=300, 
                       max_dump_size=1000,
                       chunk_size=1000000,
                       target_directory='tokenized_data'):
    
    if not os.path.exists(target_directory):
            os.makedirs(target_directory)

    # need to split the data as it won't fit my local memory
    raw_chunks = []
    CHUNK_MAX_SIZE = chunk_size
    start_idx = 0
    while start_idx < len(data):
#         print(f"Iter: {len(raw_chunks) + 1}")
        stop_idx = len(data) if start_idx + CHUNK_MAX_SIZE > len(data) else start_idx + CHUNK_MAX_SIZE
        raw_chunks.append((start_idx, stop_idx))
        start_idx = stop_idx
    print(f"Chunked the input data into {len(raw_chunks)} pieces, processing")
    metadata = {}
    file_idx = 0
    for start, stop in tqdm.tqdm(raw_chunks, desc='Tokenizing chunks'):
        tok = tokenizer.encode(data[start:stop])
        chunks = [tok[i:i + seq_length] for i in range(0, len(tok) - min_sample_length, seq_length)]
        if chunks and len(chunks[-1]) < seq_length:
            chunks[-1] += [pad_token] * (seq_length - len(chunks[-1]))
        
        print(f"Number of data samples: {len(chunks)}")
        files = [(i, i + max_dump_size) for i in range(0, len(chunks), max_dump_size)]
        print(f"DEBUG: {files=}")
        if not files:
            continue
        files[-1] = (files[-1][0], len(chunks))
        for start, stop in files:
            fname = f"dump-{file_idx}.pkl"
            file_idx += 1
            metadata[fname] = {
            'num_samples': stop - start,
            'num_tokens': (stop - start) * seq_length
            }
            with open(os.path.join(target_directory, fname), 'wb') as dump_file:
                pickle.dump(chunks[start:stop], dump_file)
    with open(os.path.join(target_directory, 'dump-meta_data.json'), 'w') as file:
        json.dump(metadata, file)
    print(f"DONE! Total number of files: {len(metadata)}, {seq_length=}")
    print(f"Total number of samples: {sum([v['num_samples'] for v in metadata.values()])}")
    print(f"Total number of tokens: {sum([v['num_tokens'] for v in metadata.values()]) / 1000000}M")
    
            



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_base', type=str, default='deepseek-ai/deepseek-coder-33b-base',
                        help='Tokenizer base name')
    parser.add_argument('--data_file', type=str, help='Path to the fetched data file')
    parser.add_argument('--tokenization_folder', type=str, help='Path to where the tokenized data will be stored')
    parser.add_argument('--min_sample_length', type=int, default=1000, help='Minimum length of a tokenized sample')
    parser.add_argument('--seq_length', type=int, default=3*512, help='Tokenized sample length')
    parser.add_argument('--max_dump_size', type=int, default=10000, help='Maximum number of samples per dump')
    parser.add_argument('--pad_token', type=int, default=32014, help='Pad token ID')
    parser.add_argument('--chars_to_tokenize', type=int, default=100000000, help='How many characters to tokenize from the input file')
    parser.add_argument('--chunk_size', type=int, default=10000000, help='To chunk the input file so that it fits in memory for tokenization')


    return parser.parse_args()




if __name__ == '__main__':
    args = parse_args()

    print(args)

    print(f"Fetching tokenizer: {args.tokenizer_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_base)

    print(f"Reading data file: {args.data_file}") 
    data_raw = pd.read_parquet(args.data_file)

    raw_code = " ".join((data_raw['content']))
    print(f"Loaded raw code, no. characters: {len(raw_code)/1e6} M")

    chunk_and_tokenize(
        data=raw_code[:args.chars_to_tokenize],
        tokenizer=tokenizer,
        pad_token=args.pad_token,
        seq_length=args.seq_length,
        min_sample_length=args.min_sample_length,
        max_dump_size=args.max_dump_size,
        chunk_size=args.chunk_size,
        target_directory=args.tokenization_folder
    )



