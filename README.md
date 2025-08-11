# Adaf-Spectrogram
We proposes a novel **Adaptive Frequency-Axis Spectrogram (Adaf-Spectrogram)**. This method uses a data-driven approach to automatically adjust the frequency axis scaling by computing the overall frequency energy distribution across an entire dataset, thereby more effectively emphasizing critical frequency features. Experimental results demonstrate that the proposed Adaf-Spectrogram exhibits excellent adaptability across multiple datasets. Furthermore, it outperforms conventional linear-scale spectrograms in recognition tasks, showcasing a significant performance improvement.
<p align="center">
  <img src="https://github.com/ding-yan/Adaf-Spectrogram/raw/main/Adaf-Spectrogram%20Process%20diagram.png" alt="Adaf-Spectrogram Process Diagram" width="700"/>
</p>

## Original vs. Adaf-Spectrogram
Here we compare the standard spectrogram with the Adaf-Spectrogram. The adaptive scaling redistributes the frequency axis according to the dataset’s energy distribution, enabling better visibility of important patterns and structures in the signal.
<p align="center">
  <img src="https://github.com/ding-yan/Adaf-Spectrogram/blob/main/Original%20vs.%20Adaf-Spectrogram.png" alt="Original vs. Adaf-Spectrogram" width="700"/>
</p>

## Environment
### Generate spectrogram
To reproduce the spectrogram generation process, please make sure to install the required Python packages with the following versions:

```bash
pip install numpy==1.24.3
pip install matplotlib==3.9.2
pip install scipy==1.8.0
pip install Pillow==8.4.0
pip install librosa==0.10.2.post1
```

## Datasets
| Abbreviation | Full Name                                | Source (Link)                                                                 |
|--------------|------------------------------------------|--------------------------------------------------------------------------------|
| MicSigV1     | Microsismic Signal Dataset Version 1     | [ESeismic - MicSigV1](https://www.igepn.edu.ec/senales-sismicas/fomulario-eseismic)   |
| ESC50        | Environmental Sound Classification 50    | [GitHub - ESC50](https://github.com/karolpiczak/ESC-50)                                |
| BSC5         | bird song data set                       | [Kaggle - BSC5](https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set/data) |
| TTSwing2.0   | Table Tennis Swing Dataset v2            | (Not publicly available due to privacy concerns) |
> **Note:**  
> The TTSwing2.0 dataset contains personal information and is therefore not publicly released, along with its corresponding code.

## Kaggle Link
- All experiments in this study were implemented using the PyTorch framework and trained on the Kaggle platform with NVIDIA P100 GPUs.  
- The training code is available at the following Kaggle notebooks, or you can download them from the respective dataset folders in this repository.

| Dataset    | Model | Kaggle Notebook Link                                                                                         |
|------------|--------|-------------------------------------------------------------------------------------------------------------|
| MicSigV1   | CNN    | [MicSigV1 - K-Fold CNN](https://www.kaggle.com/code/dingyan0418/micsigv1-k-fold-cnn)                        |
| MicSigV1   | ViT    | [MicSigV1 - 5-Fold ViT](https://www.kaggle.com/code/dingyan0418/micsigv1-5-fold-vit)                         |
| ESC50      | CNN    | [ESC50 - 5-Fold CNN](https://www.kaggle.com/code/dingyan0418/esc50-5-fold-cnn)                               |
| ESC50      | ViT    | [ESC50 - 5-Fold ViT](https://www.kaggle.com/code/dingyan0418/esc50-5-fold-vit)                               |
| BSC5       | CNN    | [BSC5 - 5-Fold CNN](https://www.kaggle.com/code/dingyan0418/bsc5-5-fold-cnn)                                 |
| BSC5       | ViT    | [BSC5 - 5-Fold ViT](https://www.kaggle.com/code/dingyan0418/bsc5-5-fold-vit)                                 |
- After clicking the link, click **"Copy & Edit"** to modify and run the notebook.
> **Note:**  
> Each Kaggle notebook is pre-configured with the required input datasets, which follow a consistent naming convention to make it easy to understand the spectrogram type and configuration used.

> **Dataset Naming Format:**
>
> ```
> <Dataset>_<SpectrogramType>_<NumberOfSections or Bands>(<SpectrumType>)
> ```
> - `<Dataset>`: Name of the dataset (e.g., MicSigV1, ESC50, BSC5)
> - `<SpectrogramType>`: Type of spectrogram used (e.g., Adaf, Mel, Spectrograms)
> - `<NumberOfSections or Bands>`: Number of frequency sections or mel bands (if applicable)
> - `<SpectrumType>`: Type of spectrum representation — Amplitude, Power, or dB (optional)

## Project Structure

```
Root/
│
├── BSC5/ # Scripts and notebooks for BSC5 dataset
│ ├── BSC5_stage1_select_audio.py   # Stage 1: Select 5422 labeled spectrograms from 9107 bird song audio clips
│ ├── BSC5_stage2_labeled.py        # Stage 2: Labeling bird species (0-4 for 5 species)
│ ├── BSC5_Ori_Spectrogram.py       # Generate Original spectrogram for BSC5 dataset
│ ├── BSC5_Mel_Spectrogram_128.py   # Generate Mel-Spectrogram (n_mels=128) for BSC5 dataset
│ ├── BSC5_Adaf_Spectrogram.py      # Generate Adaf-Spectrogram for BSC5 dataset
│ ├── bsc5-5-fold-cnn.ipynb         # CNN 5-fold cross-validation experiment
│ └── bsc5-5-fold-vit.ipynb         # Vision Transformer 5-fold cross-validation experiment
│
├── ESC50/ # Scripts and notebooks for ESC50 dataset
│ ├── ESC50_Ori_Spectrogram.py     # Generate Original spectrogram for ESC50 dataset
│ ├── ESC50_Mel_Spectrogram_128.py # Generate Mel-Spectrogram (n_mels=128) for ESC50 dataset
│ ├── ESC50_Adaf_Spectrogram.py    # Generate Adaf-Spectrogram for ESC50 dataset
│ ├── esc50-5-fold-cnn.ipynb       # CNN 5-fold cross-validation experiment
│ └── esc50-5-fold-vit.ipynb       # Vision Transformer 5-fold cross-validation experiment
│
├── MicSigV1/ # Scripts and notebooks for MicSigV1 dataset
│ ├── MicSigV1_stage1_select_and_labeled.py   # Stage 1: Select & label seismic signals (LP and VT events from BREF station)
│ ├── MicSigV1_Ori_Spectrogram.py             # Generate original spectrogram for MicSigV1 dataset
│ ├── MicSigV1_Mel_Spectrogram_64.py          # Generate Mel-Spectrogram (n_mels=64) for MicSigV1 dataset
│ ├── MicSigV1_Adaf_Spectrogram.py            # Generate Adaf-Spectrogram for MicSigV1 dataset
│ ├── micsigv1-5-fold-vit.ipynb               # Vision Transformer 5-fold cross-validation experiment
│ └── micsigv1-k-fold-cnn.ipynb               # CNN k-fold(k=5 or 10) cross-validation experiment
│
├── Adaf-Spectrogram Process diagram.png    # Workflow diagram for Adaf-Spectrogram
├── Original vs. Adaf-Spectrogram.png       # Visual comparison between conventional and Adaf-Spectrogram
└── README.md                               # Project documentation
```
> **Note:**  
> 1. Please download the dataset manually before running any scripts.  
> 2. Run the Stage scripts first (which include data selection and labeling).  
> 3. Generate the spectrograms as needed (Original, Mel, Adaf).  
>    * Please modify the spectrogram generation parameters according to your needs.  
> 4. Finally, run model training and evaluation (CNN or ViT).
