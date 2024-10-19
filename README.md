# Aerial Image Classification

Due to the large size of the datasets, they cannot be included directly in this repository. You can download the datasets from the publicly available links below.

## Publicly Available Dataset Download Links:
- **UCM**: [Link](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
- **NWPU-RESISC45**: [Link](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&cid=5C5E061130630A68&id=5C5E061130630A68%21107&parId=5C5E061130630A68%21112&action=locate)
- **PatternNet**: [Link](https://drive.google.com/file/d/127lxXYqzO6Bd0yZhvEbgIfz95HaEnr9K/view)
- **AID**: [Link](https://pan.baidu.com/s/1mifOBv6#list/path=%2FDataset)

## Dataset Directory Format:

Please ensure your dataset is organized in the following format:


## Scripts

To compute the alpha values, use the `compute_alpha.py` script with the following command:

```bash
python compute_alpha.py --dir path_to_dataset_directory
```
Run the feature extraction script to generate feature files (X.npy and Y.npy):
```bash
python feature_extractor.py --dir path_to_dataset_directory --block block_numbers_separated_by_underscore
```
Use the following command to classify the dataset:
```bash
python classify.py --nPCA number_of_pca_components --trainsize percentage_of_training_samples
```

## MIT License

Copyright (c) 2024 John Doe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

