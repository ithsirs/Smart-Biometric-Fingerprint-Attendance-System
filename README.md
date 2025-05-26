# 🔐 Smart Biometric Fingerprint Attendance System

A fully-integrated, deep learning-based fingerprint recognition and attendance system. For highly accurate biometric identification, it uses **triplet loss**, **PostgreSQL with pgvector**, and **cosine similarity**. This system offers a secure, efficient, and intelligent way to track attendance through fingerprint embeddings.

---

## 📌 Key Features

* 🧠 Fingerprint recognition using a Triplet Loss CNN model
* 📏 Generates 128-dimensional fingerprint embeddings
* 🗄️ PostgreSQL integration with `pgvector` for vector storage
* ⚡ Fast identity matching using Cosine Similarity + HNSW indexing
* 🧾 Real-time attendance logging based on biometric match
* 🧪 Evaluated using the SOCOFing dataset (55K+ samples)

---

## 🗃️ Project Structure

```
├── Biometrics_1_model_generation.ipynb     # Train Triplet Loss model and save fingerprint encoder
├── Biometrics_db_create.ipynb              # Generate embeddings and store them in PostgreSQL
├── Biometrics_3_Prediction_Update.ipynb    # Match new fingerprint and log attendance
├── Biometric_4_Addition.ipynb              # Add new user fingerprints and update PostgreSQL database
```

---

## 🛠️ Technologies Used

| Component      | Tools / Frameworks                |
| -------------- | --------------------------------- |
| Language       | Python 3.8+                       |
| Deep Learning  | PyTorch, Torchvision              |
| Database       | PostgreSQL + `pgvector` extension |
| Vector Search  | Cosine Similarity + HNSW Index    |
| Dataset        | SOCOFing                          |
| Image Handling | PIL, OpenCV                       |

---

## 🧠 Model Overview

* **Architecture**: Modified ResNet-based CNN
* **Loss Function**: TripletMarginLoss
* **Embedding Size**: 128-Dimensional vectors
* **Goal**: Minimize anchor-positive distance, maximize anchor-negative distance

---

## ⚙️ Workflow

### 🔹 1. Model Training

* Notebook: `Biometrics_1_model_generation.ipynb`
* Input: Triplets (Anchor, Positive, Negative)
* Output: `fingerprint_model.pth` (saved encoder model)

### 🔹 2. Embedding and Database Creation

* Notebook: `Biometrics_db_create.ipynb`
* Generates embeddings from images
* Stores metadata and embeddings in PostgreSQL using `pgvector`

### 🔹 3. Fingerprint Matching & Attendance Logging

* Notebook: `Biometrics_3_Prediction_Update.ipynb`
* Accepts a new fingerprint image
* Compares embedding with stored vectors
* Logs attendance if similarity score exceeds threshold

---

## 🧾 PostgreSQL Setup

Install the `pgvector` extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Database Tables:**

* `users`: Basic user info (ID, name, class)
* `embeddings`: Fingerprint vectors with HNSW index
* `attendance`: Timestamped logs with user ID and match score

---

## 🚀 Getting Started

### 🔸 Clone Repository

```bash
git clone https://github.com/yourusername/smart-biometric-attendance.git
cd smart-biometric-attendance
```

### 🔸 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔸 Setup PostgreSQL with pgvector

Run PostgreSQL locally or use cloud services like Supabase or Aiven that support `pgvector`.

---

## 📁 Dataset Information

**SOCOFing Dataset**

* 600 subjects, over 55,000 fingerprint images
* Includes real + synthetically altered fingerprints
* Download: [Kaggle SOCOFing Dataset](https://www.kaggle.com/datasets/ruizgara/socofing)

---

## 🔮 Future Enhancements

* 🌐 Web-based interface
* 📤 Bulk upload of user data 
* 🧩 Add capability to enroll new fingerprints live

---


## 📜 License

Licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
