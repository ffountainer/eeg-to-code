from features import get_raw_tensor, get_feature_tensor

class Session:
    def __init__(self, kind: str):
        self.raw = None
        self.stages = {}
        self.kind = kind
        self.tensors = {}

    def set_tensors(self):
        keys = list(self.stages.keys())
        tensor = dict.fromkeys(keys)
        for stage in tensor:
            tensor[stage] = {"raw": None, "processed": None}
            tensor[stage]["raw"] = get_raw_tensor(self.stages[stage])
            tensor[stage]["processed"] = get_feature_tensor(
                self.stages[stage], self.kind
            )
        self.tensors = tensor

    def get_tensors(self):
        return self.tensors

    def get_stages(self):
        return self.stages