# 📚 Offline Exam Malpractice Detection Tool

Welcome to the **Offline Malpractice Detection System**!  
This project helps monitor and detect suspicious activities (like head movements, eye movements) during offline exams using **computer vision** — without any internet dependency.

---

## ✨ Features

- 🎥 Real-time video monitoring with camera.
- 🦰 Intelligent detection using **OpenCV** and **MediaPipe**.
- 🚨 Alerts for malpractice behaviors like frequent head turns, hand signs.
- 📊 User-friendly GUI built with **Tkinter**.
- 🛠 Offline functionality — **no internet required**.
- 📂 Recording and saving of suspicious frames for review.

---

## 🛠 Technologies Used

- **Python**
- **OpenCV** — for real-time image processing
- **MediaPipe** — for facial and pose detection
- **Tkinter** — for GUI interface
- **Numpy** — for efficient data handling

---

## 🚀 Installation

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/your-repository-name.git
```

2. **Navigate to the project directory**  
```bash
cd your-repository-name
```

3. **Install the required libraries**  
```bash
pip install -r requirements.txt
```
*(If you don't have a `requirements.txt`, it can be created easily.)*

4. **Run the application**  
```bash
python exam_monitor.py
```

---

## 📸 How It Works

- The system captures live video from the webcam.
- It uses **MediaPipe Face Mesh** and **Pose Detection** to track head and eye movements.
- If abnormal movement (like looking away from the screen) is detected, the system raises an alert.
- Suspicious frames can be saved for later manual review.

---

## 📂 Project Structure

```plaintext
📆 Malpractice-Detection
 ├📜 malpractice_detection.py       # Core detection logic
 ├📜 violation_reviewer.py          # To view captured photos
 ├📜 exam_monitor.py                # Main file
 ├📜 requirements.txt               # Python dependencies
 ├📜 README.md                      # Project documentation
```

---

## 🎯 Future Improvements

- Add audio alerts when malpractice is detected.
- Improve detection accuracy using custom trained models.
- Generate detailed malpractice reports automatically.

---

## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you would like to change.

---


## 🙏 Acknowledgements

- **OpenCV** for image processing support.
- **MediaPipe** by Google for powerful real-time ML pipelines.
- **Tkinter** for simple and effective GUI design.

---

# 🚀 Let's make offline exams more secure together! Made with LOVE by Kirthivasan M R
