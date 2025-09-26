# 🚀 Neural Face Recognition Studio

An AI-powered facial recognition web application built using **Streamlit**, **OpenCV**, and **MediaPipe**. Designed with a modern UI and focused on **accuracy**, **speed**, and **ease-of-use**, this app allows users to **register facial identities** and **recognize faces** with real-time results and confidence scores.

---

## 🧠 Features

- 🛡️ **Secure Identity Registration**  
  Upload 2–6 high-quality facial images to register your identity with AI-powered feature extraction.

- 🔍 **AI-Powered Face Recognition**  
  Instantly analyze any photo and match it against the registered database using **ORB descriptors** and **BFMatcher**.

- 🎨 **Futuristic Glassmorphism UI**  
  Custom-designed modern interface with animated effects, gradients, and floating elements.

- 🧠 **MediaPipe + OpenCV Integration**  
  Combines Google's MediaPipe Face Detection with OpenCV's ORB for lightweight and accurate recognition.

- 🎯 **Confidence Score & Match Quality**  
  Each match includes a dynamic confidence score and quality rating for user transparency.

---

## 📦 Tech Stack

| Tool        | Purpose                             |
|-------------|-------------------------------------|
| `Python`    | Core programming language           |
| `Streamlit` | Web app framework                   |
| `OpenCV`    | Image processing & feature matching |
| `MediaPipe` | Face detection                      |
| `ORB`       | Facial feature descriptor           |
| `Pickle`    | Local data persistence              |
| `NumPy`     | Efficient array operations          |

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/neural-face-recognition-studio.git
cd neural-face-recognition-studio
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
📦 neural-face-recognition-studio
├── app.py                 # Main Streamlit app
├── face_embeddings.pkl    # Stored facial features (auto-generated)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

---

## 📷 How It Works

### 🔐 Register Identity
1. Enter a name  
2. Upload 2–6 clear face images  
3. ORB + MediaPipe extract and store facial descriptors  

### 🔎 Face Recognition
1. Upload any test image  
2. System detects the face  
3. Matches ORB descriptors with saved identities  
4. Displays result with confidence score  

---

## 🧪 Example Use Cases

- Biometric attendance systems  
- Personalized smart kiosks  
- Prototype for face-based authentication  
- Learning tool for computer vision students  

---

## 💡 Future Improvements

- 🎥 Real-time webcam support  
- ☁️ Cloud-based storage (e.g., Firebase, S3)  
- 🔒 User authentication system  
- 🧬 Switch to deep learning (e.g., FaceNet or Dlib)  
- 📈 Dashboard to visualize face embeddings and match history  

---

## ✅ Requirements

- Python 3.7+  
- Webcam or image files  
- Recommended: Run in a virtual environment  

---

## 📄 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 🙌 Acknowledgements

- MediaPipe by Google  
- OpenCV  
- Streamlit  
- JetBrains Mono & Inter Fonts  
- Gradient animation inspiration from CSS Zen Garden  

---

🌟 **Show Some Love**  
If you found this helpful, give the repo a ⭐, share it, or contribute to make it better!

> “Experience the future of facial recognition — now.”
