# YOHO-on-VOICe

Evaluating robustness of You Only Hear Once(YOHO) Algorithm on noisy audios in the VOICe Dataset ([preprint](https://arxiv.org/abs/2111.01205))

VOICe Dataset Link, click [here](https://zenodo.org/record/3514950).


Instructions:
0. Install `requirements.txt`
1. Set required parameters like `sample_rate` in `config.py`
2. Download and extract the VOICe dataset in the `data/9dB` and `data/3dB` folder inside the `VOICe/` folder
3. Run `python process_dataset.py`