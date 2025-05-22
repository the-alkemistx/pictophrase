# 📸 PictoPhrase

**PictoPhrase** is an image captioning application that generates natural language descriptions from images. You can run it either as a command‑line tool or as a web server with a sleek GUI.

---

## 🚀 Features

- **State‑of‑the‑art Model**: Uses InceptionV3 + LSTM with Bahdanau attention for accurate captions.
- **Dual Interface**:
  - **CLI** mode (`main.py`) for batch processing.
  - **Web** mode (`web.py`) for an interactive experience.
- **Speech Output**: Automatically converts generated captions to speech.
- **Easy Setup**: All dependencies managed via Conda environment.

---

## 📋 Prerequisites

- [Git](https://git-scm.com/downloads)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Python 3.8+


---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PictoPhrase.git
cd PictoPhrase
```


### 1. Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate pictophrase
```

### 2. (Optional) Install via pip

If you prefer using pip:

```bash
pip install -r requirements.txt
```

---

## 🎯 Usage

### 1. Command‑Line Interface (CLI)

Run `main.py` to generate captions for a single image or a directory of images:

```bash
# Single image
python main.py

# Directory of images
python main.py 
```

Use `-h` or `--help` for more options:

```bash
python main.py
```

---

### 2. Web Server

Run `web.py` to start the Flask web application:

```bash
# From project root ('pictophrase/')
python webapp/web.py
```

Then open your browser and navigate to `http://127.0.0.1:5000`.

Drag & drop or upload an image, and see a live caption plus audio output!

---

## 📂 Project Structure

```
pictophrase/
├── app/                   # Core application modules
│   ├── inference.py       # Caption generation logic
│   ├── preprocess.py      # Image preprocessing utilities
│   ├── feature_extractor.py
│   └── speech.py          # Text‑to‑speech
│
├── data/                  # Models & tokenizers
│   ├── best_caption_model.keras
│   └── tokenizer.pkl
│
├── webapp/                # Flask web interface
│   ├── web.py             # Flask app entry point
│   └── templates/         # HTML templates
│       └── index.html
│
├── environment.yml        # Conda environment spec
├── main.py                # CLI entry point
└── README.md              # This file
```

---

## 🤝 Contributing

Feel free to open issues or submit pull requests. Please follow the [Contributor Guidelines](CONTRIBUTING.md) if available.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## ✉️ Contact

Made with ❤️ by **Your Name** (@yourhandle).

For support or questions, open an issue or reach out at **your.email@example.com**.

