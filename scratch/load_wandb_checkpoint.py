import os
from jamun.model.denoiser_conditional import Denoiser

def load_model_from_local_checkpoint(checkpoint_dir: str):
    """
    Loads a model from a local checkpoint directory.
    """
    try:
        checkpoint_file = None
        for file_name in os.listdir(checkpoint_dir):
            if file_name.endswith(".ckpt"):
                if "last.ckpt" in file_name:
                    checkpoint_file = file_name
                    break
                checkpoint_file = file_name  # fallback to first .ckpt

        if checkpoint_file:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            print(f"Found checkpoint file: {checkpoint_path}")

            # load model
            model = Denoiser.load_from_checkpoint(checkpoint_path)
            print("Model loaded successfully!")
            print(model)

            return model
        else:
            print(f"No checkpoint file (.ckpt) found in directory: {checkpoint_dir}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    checkpoint_dir = "/data2/sules/jamun-conditional-runs/outputs/train/dev/runs/2025-06-30_19-07-58/checkpoints"
    load_model_from_local_checkpoint(checkpoint_dir) 