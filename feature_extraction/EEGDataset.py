from torch.utils.data import Dataset
from torch import Tensor

class EEGDataset(Dataset):
    def __init__(
        self,
        annotations,
        main_sessions,
        baseline_sessions,
        index_from_triple,
        index_from_idx,
    ):
        self.labels = annotations
        self.main = main_sessions
        self.baseline = baseline_sessions
        self.index_from_triple = index_from_triple
        self.index_from_idx = index_from_idx

    def __len__(self):
        return len(self.index_from_idx)

    def __getitem__(self, idx: int):

        # one input is one epoch

        session_idx, stage, epoch_idx = self.index_from_idx[idx]

        mn_tensors = self.main[session_idx].get_tensors()
        main_raw = mn_tensors[stage]["st_raw"][epoch_idx] # (n_channels, n_times)
        main_feature = mn_tensors[stage]["st_processed"][epoch_idx] # (234)

        bs_tensors = self.baseline[session_idx].get_tensors()
        bs_open = bs_tensors["eyes open"]["st_processed"] # (234)
        bs_closed = bs_tensors["eyes closed"]["st_processed"] # (234)

        data = {
            "raw": main_raw,
            "feature": main_feature,
            "eyes_open": bs_open,
            "eyes_closed": bs_closed,
        }
        label = {"stage": stage, "tokens": Tensor(self.labels[stage]["tokens"])}

        return data, label