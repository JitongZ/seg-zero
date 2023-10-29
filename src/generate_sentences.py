import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import argparse
from tqdm import tqdm
import os

def generate_captions(input_prompt, num_sentences):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    res = []
    for i in tqdm(range(num_sentences // 16)):
        outputs = model.generate(
            input_ids, temperature=0.8, num_return_sequences=16, do_sample=True, max_new_tokens=128, top_k=10
        )
        res.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return res


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="black eyes")
    parser.add_argument('--target', type=str, default="blue eyes")
    parser.add_argument('--num_sentences', type=int, default=1000)
    parser.add_argument('--output_folder', type=str, default='assets/sentences/')
    args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)


source_text = f"Provide a caption for images containing a {args.source}. "
"The captions should be in English and should be no longer than 150 characters."

target_text = f"Provide a caption for images containing a {args.target}. "
"The captions should be in English and should be no longer than 150 characters."

source_captions = generate_captions(source_text, args.num_sentences)
target_captions = generate_captions(target_text, args.num_sentences)

# import ipdb
# ipdb.set_trace()

# Check if the folder exists, if not, create it
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# Path to the file
source_file_path = os.path.join(args.output_folder, f'{args.source}.txt')
target_file_path = os.path.join(args.output_folder, f'{args.target}.txt')

# TODO: add filter to delete sentences that doesn't contain the source/target words

# Write the sentences to the file, each on a new line
with open(source_file_path, 'w') as file:
    for sentence in source_captions:
        file.write(sentence + '\n')
print(f"File {source_file_path} has been successfully saved")

with open(target_file_path, 'w') as file:
    for sentence in target_captions:
        file.write(sentence + '\n')
print(f"File {target_file_path} has been successfully saved")