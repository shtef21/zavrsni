
import torch
import os
from glob import glob
from tqdm import tqdm 
from pprint import pprint 

paths = glob('./checkpoints/*')
chk_arr = []

print('Loading checkpoints...')
for idx, path in tqdm(enumerate(paths), total=len(paths), leave=False):
  if 'best' not in path:
    continue
  chk = torch.load(path, map_location='cpu')
  chk["name"] = os.path.basename(path).replace('.chk', '')
  chk_arr.append(chk)

if len(chk_arr) == 0:
    print('No checkpoints found.')

else:
    chk_arr_sorted = sorted(chk_arr, key=lambda el: -el["accuracy"])
    print()

    chk_arr_sorted = [
    {
        'ac': format(item["accuracy"], '.2f'),
        'e': format(item["epoch"], '02d'),
        'name': item["name"],
    }
    for item in chk_arr_sorted if item["accuracy"] > 94
    ]

    pprint(chk_arr_sorted)
