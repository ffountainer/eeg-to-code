from feature_extraction.features import get_stage_raw_tensor, get_stage_feature_tensor
import torch

class Session:
    def __init__(self, kind: str):
        self.raw = None
        self.stages = {}
        self.kind = kind
        self.tensors = {}

    def set_stage_tensors(self):
        keys = list(self.stages.keys())
        tensor = dict.fromkeys(keys)
        for stage in tensor:
            tensor[stage] = {"st_raw": None, "st_processed": None}
            tensor[stage]["st_raw"] = get_stage_raw_tensor(self.stages[stage])
            tensor[stage]["st_processed"] = get_stage_feature_tensor(
                self.stages[stage], self.kind
            )
        self.tensors = tensor

    def get_tensors(self):
        return self.tensors

    def get_stages(self):
        return self.stages

    # we get a dict ["sess_raw", "st_features"],
    # where sess_raw is a tensor of raw data for a session
    # and st_features is a dictionary that contains "st_raw" and "st_processed" tensors for each stage of a session
    def get_session_features(self) -> dict[str, str]:
        features = dict.fromkeys("sess_raw", "st_features") # st for stages
        features["sess_raw"] = torch.from_numpy(self.raw.get_data()).float()
        features["st_features"] = self.tensors
        return features