# Adaf-Spectrogram
We proposes a novel **Adaptive Frequency-Axis Spectrogram (Adaf-Spectrogram)**. This method uses a data-driven approach to automatically adjust the frequency axis scaling by computing the overall frequency energy distribution across an entire dataset, thereby more effectively emphasizing critical frequency features. Experimental results demonstrate that the proposed Adaf-Spectrogram exhibits excellent adaptability across multiple datasets. Furthermore, it outperforms conventional linear-scale spectrograms in recognition tasks, showcasing a significant performance improvement.
<p align="center">
  <img src="https://github.com/ding-yan/Adaf-Spectrogram/raw/main/Adaf-Spectrogram%20Process%20diagram.png" alt="Adaf-Spectrogram Process Diagram" width="700"/>
</p>

## Environment
creat Adaf-Spectrogram:

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
> **Note:**  
> Each Kaggle notebook is pre-configured with the required input datasets.  
> 
> The dataset image folders follow a consistent naming convention, making it easy to understand the spectrogram type and configuration used.

> **Folder Naming Format:**
>
> ```
> <Dataset>_<SpectrogramType>_<NumberOfSections or Bands>(<SpectrumType>)
> ```
> - `<Dataset>`: Name of the dataset (e.g., MicSigV1, ESC50, BSC5)
> - `<SpectrogramType>`: Type of spectrogram used (e.g., Adaf, Mel, Spectrograms)
> - `<NumberOfSections or Bands>`: Number of frequency sections or mel bands (if applicable)
> - `<SpectrumType>`: Type of spectrum representation — Amplitude, Power, or dB (optional)



