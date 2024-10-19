
# A Lightweight Relu-based Feature Fusion for Aerial Scene Classification

Due to the large size of the datasets, they cannot be included directly in this repository. You can download the datasets from the publicly available links below.

## Publicly Available Dataset Download Links:
- **UCM**: [Link](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
- **NWPU-RESISC45**: [Link](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&cid=5C5E061130630A68&id=5C5E061130630A68%21107&parId=5C5E061130630A68%21112&action=locate)
- **PatternNet**: [Link](https://drive.google.com/file/d/127lxXYqzO6Bd0yZhvEbgIfz95HaEnr9K/view)
- **AID**: [Link](https://pan.baidu.com/s/1mifOBv6#list/path=%2FDataset)

## Dataset Directory Format:

Please ensure your dataset is organized in the following format:
Dataset Directory Format:
Example:
AID/
---Airport/
----------airport_1.jpg
----------airport_2.jpg
---BareLand/
-----------bareland_1.jpg
-----------bareland_2.jpg
.....

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


## References

```bibtex
@inproceedings{arefeen2021lightweight,
  title={A lightweight relu-based feature fusion for aerial scene classification},
  author={Arefeen, Md Adnan and Nimi, Sumaiya Tabassum and Uddin, Md Yusuf Sarwar and Li, Zhu},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={3857--3861},
  year={2021},
  organization={IEEE}
}




