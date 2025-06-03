import torch
import os

# === Configuration ===
# Path to your full checkpoint (with optimizer state)
input_checkpoint_path = "checkpoint/vits_ljspeech-May-31-2025_09+16PM-4128b3b/best_model.pth"

# Path to save the optimized, smaller checkpoint
output_model_path = "checkpoint/vits_ljspeech-May-31-2025_09+16PM-4128b3b/optimized_model.pth"

# === Load checkpoint ===
print(f"Loading checkpoint: {input_checkpoint_path}")
checkpoint = torch.load(input_checkpoint_path, map_location="cpu")

# === Check contents ===
print(f"Checkpoint keys: {list(checkpoint.keys())}")

# === Remove optimizer state if present ===
if 'optimizer' in checkpoint:
    print("Removing optimizer state...")
    del checkpoint['optimizer']
else:
    print("No optimizer state found. Nothing to remove.")

# Save the checkpoint without deleted optimizer. 
# In xtts.py of coqui-ai-TTS the keys will auto ignored ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
torch.save(checkpoint, output_model_path)
print(f"Saved shrinked model to: {output_model_path}")


# Optional: Show size reduction
orig_size = os.path.getsize(input_checkpoint_path) / (1024 * 1024)
new_size = os.path.getsize(output_model_path) / (1024 * 1024)
print(f"Original size: {orig_size:.2f} MB -> Optimized size: {new_size:.2f} MB")

