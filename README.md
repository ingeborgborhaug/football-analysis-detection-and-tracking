# Remote Access & GPU Usage Guide (NTNU)

## VPN Access (If You're Off-Campus)

To connect to NTNU resources from outside the local network, you **must** be connected to the NTNU VPN.

📌 [Install VPN – NTNU Wiki](https://i.ntnu.no/wiki/-/wiki/English/Install+vpn)

---

## Remote Access to Cybele Computers via SSH

You can access the Cybele nodes remotely using SSH. Use the following command in your terminal:

```bash
ssh <USERNAME>@clab<01-25>.idi.ntnu.no
```

- Replace `<USERNAME>` with your NTNU username.
- Replace `<01-25>` with a number from 01 to 25 (e.g., `clab03`).

###: Use VS Code's Remote - SSH Extension

You can also connect via the **Remote - SSH** extension in VS Code using the same SSH command.

---

## Usage Restrictions

- **Do not use the nodes during school hours:**  
  **08:00 – 18:00**

- **Always check that no one else is using the GPU before starting training**

### Check GPU usage:

Use the following command:

```bash
nvidia-smi
```

Make sure GPU utilization is **below 10%** before using it. Some minor background GPU processes (like Xorg or Gnome Shell) are expected.

Example output when GPU is idle:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07   Driver Version: 535.161.07   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------|
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap| Memory-Usage         | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce RTX 4090  | 00000000:01:00.0 Off |                  Off |
|  0%   31C    P8     9W / 450W | 111MiB / 24564MiB     |   0%       Default   |
+-------------------------------+----------------------+----------------------+
| Processes:                                                            GPU Mem |
|  PID   Type   Name                                                  Usage     |
|===============================================================================|
| 1650   G      /usr/lib/xorg/Xorg                                       85MiB  |
| 2140   G      /usr/bin/gnome-shell                                     16MiB  |
+-----------------------------------------------------------------------------+
```

---

## Python Environment Setup

## 1. Go into workspace:

```bash
cd /work/imborhau
```

### 2. Create a virtual environment:

```bash
python3 -m venv .venv
```

### 3. Activate the environment:

```bash
source /work/imborhau/.venv/bin/activate
```

> Adjust the path if your `.venv` is located somewhere else.

## 4. Clone repository into workspace:

```bash
git clone https://github.com/ingeborgborhaug/football-analysis-detection-and-tracking.git
```

### 5. Install dependencies from `requirements.txt`:

```bash
pip install -r football-analysis-detection-and-tracking/requirements.txt
```

---

## Run code:

```bash
python3 football-analysis-detection-and-tracking/detection.py
```
