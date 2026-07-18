import torch
import mne_features
import mne

freq_bands = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "alpha": [8, 13],
    "beta": [13, 30],
    "gamma": [30, 100],}

def get_raw_tensor(windows : mne.Epochs) -> torch.Tensor:
    tensor = windows.get_data(copy=True) # (n_epochs, n_channels, n_times)
    return torch.tensor(tensor)

def get_spectral_tensor(windows : mne.Epochs) -> torch.Tensor:
    data = windows.get_data(copy=True)
    freq = windows.info["sfreq"]
    tensor = mne_features.feature_extraction.extract_features(X=data, sfreq=freq, selected_funcs=["pow_freq_bands"],
                                                      funcs_params={"pow_freq_bands__freq_bands" : freq_bands,
                                                                     "pow_freq_bands__freq_bands" : True})
    return torch.tensor(tensor)