# Benchmarking Deep Learning Architectures for Super-Resolution of Solar Images

![Solar Super-Resolution](https://your-image-link.com)

This repository contains the code for the bachelor thesis **"Benchmarking Deep Learning Architectures for Super-Resolution of Solar Images"**. It implements and evaluates **CNNs (RRDB), GANs (ESRGAN), and Diffusion Models (LDM, ELDM)** for enhancing extreme ultraviolet (EUV) solar images captured at the **171 Å wavelength**.

## 🚀 Overview
Super-resolution of solar images is crucial for studying fine-scale structures in the solar corona. This project explores deep learning models to enhance low-resolution (LR) solar images into high-resolution (HR) outputs. The models are evaluated using **SSIM, PSNR, and a novel Statistical Fidelity (SF) metric**.

## 📂 Repository Structure
```
📂 super_resolution_solar
├── 📂 datasets              # Preprocessed solar image dataset
├── 📂 models                # Implementations of RRDB, ESRGAN, LDM, ELDM
├── 📂 scripts               # Training and evaluation scripts
├── 📂 results               # Model outputs and performance metrics
├── 📜 requirements.txt      # Python dependencies
├── 📜 train.py              # Training script
├── 📜 evaluate.py           # Evaluation script
├── 📜 README.md             # Project documentation
```

## 📊 Model Comparison
| Model         | SSIM ↑  | PSNR (dB) ↑ | SF ↓  |
|--------------|--------|------------|------|
| RRDB (CNN)  | 0.9771 | 39.91      | 0.0192 |
| ESRGAN (GAN) | 0.9735 | 41.03      | 0.00383 |
| LDM          | 0.9641 | 37.75      | 0.00749 |
| ELDM (Ours)  | 0.9687 | 38.45      | 0.00512 |

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/super_resolution_solar.git
cd super_resolution_solar

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 🏋️‍♂️ Training the Models
To train a model, run:
```bash
python train.py --model ESRGAN --epochs 150 --batch_size 8
```
Available models: `RRDB`, `ESRGAN`, `LDM`, `ELDM`

## 🧪 Evaluating Performance
Run evaluation on test images:
```bash
python evaluate.py --model ESRGAN
```

## 📊 Results & Visualization
To generate qualitative comparisons:
```bash
python visualize_results.py --input example.png
```

## 🔬 Dataset
The dataset consists of **5,000 solar images** from NASA’s **Solar Dynamics Observatory (SDO)** at 171 Å. Images are preprocessed with:
- **Downsampling** from 4096×4096 to 512×512
- **Instrumental corrections** (exposure, degradation adjustments)
- **Normalization** to the [-1, 1] range

## 🛠️ Model Details
### 1️⃣ **CNN (RRDB)**
- Uses **Residual-in-Residual Dense Blocks (RRDB)**
- Optimized with **L1 Loss** (pixel-wise similarity)

### 2️⃣ **ESRGAN (GAN)**
- Uses a **generator-discriminator framework**
- **Adversarial loss + perceptual loss** for texture enhancement

### 3️⃣ **Latent Diffusion Model (LDM)**
- Performs super-resolution in a **latent space** to reduce computation
- Uses a **U-Net** for denoising

### 4️⃣ **Enhanced Latent Diffusion Model (ELDM)** (Ours)
- Introduces **GAN-based refinement** after the latent diffusion process
- Uses **Pseudo Numerical Diffusion Methods (PNDM)** for faster inference

## 📌 Future Work
- Expand dataset to **50,000 images** from multiple EUV wavelengths
- Improve efficiency via **multi-GPU training**
- Develop **hybrid GAN-Diffusion models** for stability + texture enhancement

## 📜 Citation
If you use this code, please cite:
```
@article{elsheikh2025superresolution,
  title={Benchmarking Deep Learning Architectures for Super-Resolution of Solar Images},
  author={Mohamed Hisham Mahmoud Elsheikh},
  journal={Bachelor Thesis},
  year={2025}
}
```

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
📧 For questions, feel free to open an issue or contact [your email].
