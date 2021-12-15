def load_submission(groupid, grid=None):
    if groupid == "goudarzi":
        import torch
        from submissions.goudarzi.MobileNetV2 import MobileNetV2

        weights_file = "submissions/goudarzi/model_weights.pt"
        model = MobileNetV2()
        model.load_state_dict(torch.load(weights_file))

    elif groupid == "rothlubbers":
        import torch
        from submissions.rothlubbers.task1_bfFinal_CSW2D import Task1_bf_final_CSW2D

        weights_file = "submissions/rothlubbers/task1_bfFinal_CSW2D_stateDict.pth"
        model = Task1_bf_final_CSW2D()
        model.load_state_dict(torch.load(weights_file))

    return model


if __name__ == "__main__":
    # Example usage
    load_submission("goudarzi")
