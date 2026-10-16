# CivicSense AI 
### An Intelligent Pipeline for Automated Road Damage Detection, Prioritization, Routing, and Resolution Verification

CivicSense AI is an AI-powered civic technology project designed to detect road damage from images, prioritize complaints based on urgency, identify the responsible civic department, and verify whether reported road damage has been resolved.

##  Problem Statement

Road damage such as potholes and cracks can create safety risks and disrupt transportation. Manual complaint handling may lead to inconsistent prioritization, incorrect department routing, delayed responses, and difficulties verifying completed repairs.

CivicSense AI aims to support this process through automated road-damage analysis, urgency scoring, jurisdiction checking, knowledge-based routing, and visual resolution verification.

##  Project Objectives

- Detect potholes and different types of road cracks from road images.
- Calculate a weighted urgency score for each complaint.
- Check whether a reported location falls within GNIDA jurisdiction.
- Retrieve relevant civic information to determine the appropriate department and escalation level.
- Generate a reviewable complaint draft.
- Compare before-and-after images to assist resolution verification.
- Provide a dashboard for citizens and administrators to review complaint information.

##  Key Features

### 1. Road Damage Detection
Uses **YOLOv8**, a CNN-based deep-learning object detector, to identify road damage in uploaded images.

Supported target classes:

- Pothole
- Longitudinal crack
- Transverse crack
- Alligator crack

The detection output includes class labels, bounding boxes, and confidence scores.

### 2. Intelligent Complaint Prioritization

Each complaint receives an urgency score between 0 and 1.

| Factor | Weight |
|---|---:|
| Damage severity | 30% |
| Detection confidence | 15% |
| Similar-report frequency | 25% |
| Proximity to critical zones | 20% |
| Time pending | 10% |

The report-frequency contribution is capped at 10 similar reports, while the pending-time contribution is capped at 15 days.

**Priority score:**

`0.30 × Severity + 0.15 × Confidence + 0.25 × Frequency + 0.20 × Critical Zone + 0.10 × Pending Time`

The individual factors are normalized before the weighted score is calculated.

### 3. GNIDA Jurisdiction Verification

The system checks the reported coordinates against the configured GNIDA boundary data.

- Locations inside the configured boundary proceed to routing.
- Locations outside the boundary are explicitly flagged.
- Unavailable or inconclusive jurisdiction information can be sent for manual review.

### 4. RAG-Based Department Routing

The routing module uses Retrieval-Augmented Generation-style retrieval to find relevant information from a curated GNIDA knowledge corpus.

Technologies include:

- Sentence Transformers for text embeddings
- FAISS for similarity-based retrieval
- Curated GNIDA department and escalation information

The retrieved information supports the department and escalation decision. The system generates a complaint draft for human review rather than automatically submitting it to a live government portal.

### 5. Road Repair Verification

The verification module compares the original road image with an after-resolution image.

It uses:

- YOLOv8 re-detection
- Structural Similarity Index Measure (SSIM)
- Before-and-after detection comparison

The system reports a verification outcome or flags inconclusive cases for manual review. SSIM alone is not treated as proof that a repair is complete.

##  System Architecture

```text
Citizen / Admin Dashboard
          |
          v
     FastAPI Backend
          |
          v
   Road Image + Location
          |
          v
   YOLOv8 Road Damage Detector
          |
          v
  Class + Bounding Box + Confidence
          |
          v
    Priority Scoring
          |
          v
  RAG-Based Department Routing
   Sentence Transformers + FAISS
          |
          v
  Department + Escalation Level
          |
          v
  Reviewable Complaint Draft
          |
          v
  After-Resolution Image Upload
          |
          v
  Re-detection + SSIM Verification
          |
          v
 Verified Resolved / Manual Review
```

Model training is performed separately from normal application inference.

##  Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch |
| Object detection | Ultralytics YOLOv8 |
| Computer vision | OpenCV |
| Image similarity | scikit-image (SSIM) |
| Embeddings | Sentence Transformers |
| Vector search | FAISS |
| Backend API | FastAPI |
| Frontend dashboard | Streamlit |
| Database | SQLite / PostgreSQL |
| Data processing | NumPy, Pandas |
| Geospatial boundary data | GeoJSON |
| Model training hardware | NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM |

##  Project Structure

```text
CivicSense-AI/
├── backend/
│   ├── models/
│   ├── routes/
│   ├── services/
│   ├── config.py
│   ├── database.py
│   └── main.py
├── data/
│   ├── gnida/
│   │   ├── boundaries/
│   │   └── corpus/
│   ├── rdd2022/
│   └── test/
│       ├── before/
│       └── after/
├── database/
├── frontend/
│   ├── components/
│   ├── pages/
│   └── app.py
├── ml/
│   ├── detection/
│   ├── prioritization/
│   ├── routing/
│   └── verification/
├── models/
│   └── yolov8/
├── notebooks/
├── tests/
├── uploads/
├── vector_store/
├── requirements.txt
├── run.py
└── README.md
```

Generated training outputs, datasets, and other local files may exist outside this abbreviated structure.

##  Installation and Setup

### Prerequisites

- Python 3.13, subject to compatibility with the installed dependencies
- NVIDIA GPU and a compatible NVIDIA driver for CUDA-enabled training
- Git, if cloning the repository
- Sufficient disk space for datasets
