# GNU Radio Flowgraphs for SDR Physical Intrusion

This directory contains the GNU Radio Companion (`.grc`) flowgraphs needed to interact with the physical hardware (USRP radios) to both **generate raw datasets** and **visualize intrusion spikes**.

---

## 1. How to Generate a Dataset Using GNU Radio Hardware

To train the Machine Learning (ML) models on real-world RF data instead of the Python simulator, you must capture the physical micro-Doppler signals using your SDR hardware.

**Flowgraph to use:** `usrp_data_collector.grc`

1. **Hardware Setup:** Connect your USRP B210/X310 via USB/Ethernet. Place the TX (Transmit) and RX (Receive) antennas across a room from each other (e.g., across a hallway or doorway) to establish an invisible Line-of-Sight "tripwire."
2. **Open the Flowgraph:** Open `usrp_data_collector.grc` in GNU Radio Companion.
3. **Configure the Interface:** 
   - Ensure the Sample Rate `1.92M` and Center Frequency `2.4 GHz` are correct.
   - Adjust `TX Gain` and `RX Gain` so the waterfall isn't clipping but shows a solid wave.
4. **Begin Recording:**
   - On the GUI, type the name of the target inside the `Target Class Name` box (e.g., `"human"`).
   - Have a human walk back and forth through the invisible radio path.
   - Click the `RECORDING` radio button. 
   - A raw `human.dat` file will be generated and dropped directly into your `../dataset/` directory.
   - Disable recording, change the text box to `"animal"` (have a dog walk through) or `"drone"` (hover a drone in the room) or `"clear"` (empty room), and repeat.

### Core Components & Their Importance:
* **UHD: USRP Source / Sink:** Transmits a continuous QPSK/CW tone and receives the scattered, fading reflections simultaneously.
* **Complex to Mag:** Computes the raw amplitude from the $I + jQ$ baseband complex floats. This is crucial because physical bodies passing through cause amplitude fading.
* **Peak Detector:** A basic algorithm used as your non-ML "Burglar Alarm". It flags sharp spikes in the amplitude to trigger a beep on the time sink.
* **Waterfall Sink:** A real-time spectrogram. Crucial for verifying that the RF wave is actually picking up micro-Doppler frequencies (e.g., 2Hz arm swinging) before recording data.
* **File Sink:** Outputs the recorded baseband directly into a 32-bit complex `.dat` file required by the ML PyTorch DataLoaders.

---

## 2. Viewing Saved Datasets

If you have already generated datasets via the hardware or the Python simulator (`sim_system.py`) and just want to look at them visually:

**Flowgraph to use:** `dataset_playback.grc`

1. Open `dataset_playback.grc` and hit Play.
2. It uses a **File Source** block coupled with a **Throttle** block. This simulates a real radio receiver by unpacking the `.dat` array and slowly feeding it to the GUI exactly at 1.92 MSps. 
3. Switch between `Clear`, `Human`, and `Drone` using the UI buttons to visually see why the CNN algorithms are so effective at classifying the different spike profiles.

---

## 3. How to Run the ML Algorithm Finally on That Data

Once you have successfully generated your four real `.dat` files (`clear.dat`, `human.dat`, `animal.dat`, `drone.dat`) and they are located in `../dataset/` or `../dataset/simulated/`:

You do **not** need to use GNURadio anymore. Open your terminal in the root project directory and run the Deep Learning ablation script, bypassing the simulator hook using `--skip_data`:

```bash
# 1. Navigate to the project root
cd ../

# 2. Run the Machine Learning suite over your Hardware .dat files
python -m src.run_ablation --skip_data
```

The script will automatically ingest your `.dat` files from GNU Radio, slice them into 256-sample chunks, extract the STFT micro-Doppler spectrograms, and begin training the **Dual-Branch Fusion CNN** and your **Custom DenseNet**!
