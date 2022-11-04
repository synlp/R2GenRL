# R2GenRL
The implementation for our ACL-2022 paper titled [Reinforced Cross-modal Alignment for Radiology Report Generation](https://aclanthology.org/2022.findings-acl.38/)

## Citation

```
@inproceedings{qin-song-2022-reinforced,
    title = "Reinforced Cross-modal Alignment for Radiology Report Generation",
    author = "Qin, Han and Song, Yan",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    pages = "448--458",
}
```

## Requirements
Our code works with the following environment.
- `torch==1.5.1`
- `torchvision==0.6.1`
- `opencv-python==4.4.0.42`

Clone the evaluation tools from the [website](https://github.com/salaniz/pycocoevalcap).

## Datasets
We use two datasets (`IU X-Ray` and `MIMIC-CXR`) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://openi.nlm.nih.gov/) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/) and then put the files in `data/mimic_cxr`.


## Running
For `IU X-Ray`,
* `bash scripts/iu_xray/run.sh` to train the `Base+cmn` model on `IU X-Ray`.
* `bash scripts/iu_xray/run_rl.sh` to train the `Base+cmn+rl` model on `IU X-Ray`.

For `MIMIC-CXR`,
* `bash scripts/mimic_cxr/run.sh` to train the `Base+cmn` model on `MIMIC-CXR`.
* `bash scripts/mimic_cxr/run_rl.sh` to train the `Base+cmn+rl` model on `MIMIC-CXR`.

## Attention Plots

Change the ```path``` (line:183) variable in ```help.py``` to the image that you wish to plot and then run the script ```plot.sh```.

