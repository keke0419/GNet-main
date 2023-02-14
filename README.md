# GNet-main

An integrated context-aware neural framework for transcription factor binding signal at single nucleotide resolution prediction

## Requirements

- Pytorch 1.1 [[Install]](https://pytorch.org/)
- CUDA 10.0
- Python 3.6
- Python packages: biopython, scikit-learn, pandas, pyBigWig, scipy, matplotlib, seaborn
- Download [hg38.fa, mm10.fa](https://hgdownload.soe.ucsc.edu/downloads.html) then unzip them and put them into `Genome/`

## Data Preparation

Sequence preprocessing and one-hot coding:
```
python bedsignal-hg38.py -d <> -n <> -s <>
```

| Arguments   | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| -d          | The path of datasets, e.g. /your_path/GNet-main/HeLa-S3/CTCF   |
| -n          | The name of the specified dataset, e.g. CTCF                   |
| -s          | Random seed (default is 666)                                   |
