import zmq
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from scipy import signal as sig
import sys
import os

# Add src path so we can import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import get_model
from src.data_loader import CLASS_NAMES

# ============================================================
# CONFIGURATION
# ============================================================
ZMQ_URL = "tcp://127.0.0.1:5555"      # Must match GNU Radio ZMQ PUB Sink address
SAMPLE_RATE = 1.92e6                  # USRP Sample Rate
WINDOW_SIZE = 1280                    # Size of IQ chunk to process
MODEL_PATH = "../checkpoints/best_resnet50.pth"

# ============================================================
# LIVE INFERENCE PIPELINE
# ============================================================
def spectrogram_to_tensor(spec_data):
    """Convert raw STFT to 3-channel normalized PyTorch Tensor"""
    with np.errstate(divide='ignore', invalid='ignore'):
        spec_db = 10 * np.log10(np.abs(spec_data) + 1e-12)
    
    smin, smax = spec_db.min(), spec_db.max()
    if smax - smin < 1e-6:
        # Empty signal
        spec_norm = np.zeros_like(spec_db, dtype=np.uint8)
    else:
        spec_norm = ((spec_db - smin) / (smax - smin) * 255).astype(np.uint8)
    
    img = Image.fromarray(spec_norm, mode='L')
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(img).unsqueeze(0)  # Add batch dimension [1, 3, 224, 224]

def main():
    print(f"Loading Intrusion Detection Model from {MODEL_PATH}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model('resnet50')
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[OK] Checkpoint Loaded.")
    else:
        print("[WARN] Checkpoint not found! Running with untrained weights (for testing only).")
    
    model.to(device)
    model.eval()

    print(f"Connecting to GNU Radio ZMQ Socket at {ZMQ_URL}...")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(ZMQ_URL)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to everything
    
    print("\n" + "="*50)
    print("LIVE RF INTRUSION DETECTION STARTED")
    print("="*50 + "\n")

    iq_buffer = np.array([], dtype=np.complex64)

    while True:
        try:
            # Receive raw binary I/Q data from GNU Radio
            raw_msg = socket.recv()
            
            # GNU Radio sends complex64 (interleaved float32 I and Q)
            iq_chunk = np.frombuffer(raw_msg, dtype=np.complex64)
            iq_buffer = np.concatenate((iq_buffer, iq_chunk))
            
            # Process when we have enough data for a window
            if len(iq_buffer) >= WINDOW_SIZE:
                window = iq_buffer[:WINDOW_SIZE]
                iq_buffer = iq_buffer[WINDOW_SIZE:] # Keep remainder
                
                # Compute Spectrogram
                f, t, Sxx = sig.spectrogram(
                    window, fs=SAMPLE_RATE, nperseg=64, noverlap=48, nfft=128, return_onesided=False
                )
                
                # Convert to Tensor
                img_tensor = spectrogram_to_tensor(Sxx).to(device)
                
                # Live Inference
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    confidence, predicted_idx = torch.max(probabilities, 0)
                    
                    predicted_class = CLASS_NAMES[predicted_idx.item()]
                    conf_pct = confidence.item() * 100
                    
                    if predicted_class == 'Benign':
                        print(f"[{conf_pct:.1f}%] Status: Clear (Background Noise)")
                    elif 'Jam' in predicted_class:
                        print(f"[{conf_pct:.1f}%] \033[91mWARNING: {predicted_class} Attack Detected!\033[0m")
                    else:
                        print(f"[{conf_pct:.1f}%] \033[93mALERT: {predicted_class} Intrusion Detected!\033[0m")
                        
        except KeyboardInterrupt:
            print("\nLive Inference Stopped.")
            break
        except Exception as e:
            print(f"Error processing stream: {e}")

if __name__ == "__main__":
    main()
