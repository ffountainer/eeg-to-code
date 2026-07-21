import torch
import mne_features
import mne

freq_bands = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "alpha": [8, 13],
    "beta": [13, 30],
    "gamma": [30, 45],
}


def get_raw_tensor(windows: mne.Epochs) -> torch.Tensor:
    tensor = windows.get_data()  # (n_epochs, n_channels, n_times)
    return torch.tensor(tensor)


def get_feature_tensor(windows: mne.Epochs, kind: str) -> torch.Tensor:
    data = windows.get_data(copy=True)
    freq = windows.info["sfreq"]

    power_tensor = mne_features.feature_extraction.extract_features(
        X=data,
        sfreq=freq,
        selected_funcs=["pow_freq_bands"],
        funcs_params={
            "pow_freq_bands__freq_bands": freq_bands,
            "pow_freq_bands__normalize": True,
        },
    )
    fractal_dim_tensor = mne_features.feature_extraction.extract_features(
        X=data, sfreq=freq, selected_funcs=["higuchi_fd"]
    )
    entropy_tensor = mne_features.feature_extraction.extract_features(
        X=data,
        sfreq=freq,
        selected_funcs=["spect_entropy"],
        funcs_params={"spect_entropy__psd_method": "fft"},
    )
    wavelet_tensor = mne_features.feature_extraction.extract_features(
        X=data, sfreq=freq, selected_funcs=["wavelet_coef_energy"]
    )
    # print("Feature tensors shapes:")
    # print("Power Spectrum: ", power_tensor.shape)
    # print("Higuchi Fractal Dimension: ", fractal_dim_tensor.shape)
    # print("Spectral Entropy: ", entropy_tensor.shape)
    # print("Energy of Wavelet decomposition coefficients: ", wavelet_tensor.shape)
    power_tensor = torch.from_numpy(power_tensor)
    fractal_dim_tensor = torch.from_numpy(fractal_dim_tensor)
    entropy_tensor = torch.from_numpy(entropy_tensor)
    wavelet_tensor = torch.from_numpy(wavelet_tensor)
    tensor = torch.cat(
        [power_tensor, fractal_dim_tensor, entropy_tensor, wavelet_tensor], dim=1
    )
    if kind == "bs":
        tensor = tensor.mean(dim=0)
        return tensor
    if kind != "mn":
        raise Exception("That is not a valid session type, pal!")
    return tensor
