# 🚀 ActionSense AI

> AI-powered image analysis dashboard with deep learning action recognition and automated caption generation

![Next.js](https://img.shields.io/badge/Next.js-16.1-black?style=flat-square&logo=next.js)
![React](https://img.shields.io/badge/React-19.2-61DAFB?style=flat-square&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=flat-square&logo=typescript)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python)

ActionSense AI is a full-stack web application that combines a modern Next.js frontend with a powerful Python deep learning backend to perform real-time image analysis, action recognition, and automated caption generation.

## ✨ Features

- 🖼️ **Image Upload & Preview** - Drag-and-drop interface with instant image preview
- 🤖 **Action Recognition** - Identifies actions/objects using MobileNetV2 (ImageNet-trained)
- 📝 **Image Captioning** - Generates natural language descriptions using custom CNN + LSTM model
- 🎯 **Confidence Scoring** - Displays prediction confidence with visual progress bars
- 🎨 **Modern UI** - Beautiful, responsive design with Tailwind CSS and smooth animations
- ⚡ **Real-time Processing** - Fast inference with optimized model architecture
- 🔒 **Type-safe** - Full TypeScript support for robust development

## 🏗️ Architecture

### Frontend (Next.js + React)
- **Framework**: Next.js 16 with App Router
- **UI**: React 19 with Tailwind CSS 4
- **Icons**: Lucide React
- **Features**: Client-side image preview, form handling, API integration

### Backend (Next.js API Routes)
- **API**: RESTful endpoint at `/api/analyze`
- **Processing**: Spawns Python subprocess for ML inference
- **Validation**: File type, size, and format checking
- **Security**: 10MB file size limit, allowed image types only

### Deep Learning Model (Python + TensorFlow)
- **EfficientNetB0**: Spatial feature extraction
- **MobileNetV2**: Action classification (1000 ImageNet classes)
- **Custom Caption Model**: 
  - Bahdanau Attention mechanism
  - Bidirectional LSTM decoder
  - Beam search for optimal caption generation

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Next.js, React, TypeScript, Tailwind CSS |
| **Backend** | Node.js, Next.js API Routes |
| **ML/AI** | TensorFlow, Keras, NumPy, Pillow |
| **Models** | EfficientNetB0, MobileNetV2, Custom LSTM |
| **Styling** | Tailwind CSS, CSS Modules, Custom Animations |

## 📁 Project Structure

```
dl-dashboard/
├── app/
│   ├── api/
│   │   └── analyze/
│   │       └── route.ts          # Backend API endpoint
│   ├── globals.css               # Global styles
│   ├── layout.tsx                # Root layout
│   └── page.tsx                  # Main frontend component
├── python_model/
│   ├── predict.py                # ML inference script
│   ├── requirements.txt          # Python dependencies
│   ├── tokenizer.pkl            # Trained tokenizer (required)
│   └── caption_trained_model.h5 # Model weights (required)
├── public/
│   └── (static assets)
├── model_AC.ipynb               # Model training notebook
├── package.json                 # Node.js dependencies
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** 20.x or higher
- **Python** 3.8 or higher
- **npm** or **yarn**

### 1. Clone the Repository

```bash
git clone https://github.com/MoizBhatti333-code/ActionSenseAi.git
cd ActionSenseAi
```

### 2. Install Dependencies

**Frontend:**
```bash
npm install
# or
yarn install
```

**Python Environment:**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python packages
pip install -r python_model/requirements.txt
```

### 3. Setup Model Files

⚠️ **Required**: You need to add the trained model files to the `python_model/` directory:

1. **`tokenizer.pkl`** - Tokenizer from your training
2. **`caption_trained_model.h5`** - Trained model weights

These files are excluded from git due to their size. You can:
- Train your own model using `model_AC.ipynb`
- Download pre-trained weights (if available)
- Contact the repository owner for model files

### 4. Run the Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📖 Usage

1. **Upload an Image**: Click or drag an image into the upload area
2. **Run Analysis**: Click "Run Model Analysis" button
3. **View Results**: 
   - See the detected action class
   - View confidence scores
   - Read generated image caption
   - See top 3 predictions

## 🔌 API Documentation

### POST `/api/analyze`

Analyzes an uploaded image and returns action classification and caption.

**Request:**
```typescript
Content-Type: multipart/form-data

{
  image: File  // JPEG, PNG, or WebP (max 10MB)
}
```

**Response:**
```typescript
{
  "action_class": string,      // Primary detected action/object
  "confidence": string,        // Confidence percentage
  "annotations": [
    {
      "time": string,          // Label (e.g., "Caption", "Prediction 1")
      "text": string           // Description or prediction
    }
  ]
}
```

**Example:**
```json
{
  "action_class": "Golden Retriever",
  "confidence": "92.5%",
  "annotations": [
    {"time": "Caption", "text": "A dog is playing in the grass."},
    {"time": "Prediction 1", "text": "Golden Retriever (92.5% confidence)"},
    {"time": "Prediction 2", "text": "Labrador Retriever (5.3% confidence)"},
    {"time": "Prediction 3", "text": "Dog (1.2% confidence)"}
  ]
}
```

## 🧪 Testing the Python Model

Test the model independently:

```bash
source .venv/bin/activate
python python_model/predict.py /path/to/test/image.jpg
```

## 🎨 Model Architecture

```
Input Image (224×224×3)
    ↓
EfficientNetB0 → Spatial Features (7×7×1280)
    ↓
MobileNetV2 → Action Features (1280)
    ↓
Caption Model:
  - Embedding Layer (512D)
  - Bidirectional LSTM
  - Bahdanau Attention
  - Dense Layers
    ↓
Beam Search Decoder
    ↓
Generated Caption
```

## 🔧 Configuration

Update Python path in `app/api/analyze/route.ts` if needed:

```typescript
// Line ~120
const pythonPath = join(process.cwd(), '.venv', 'bin', 'python');
```

## 📦 Build for Production

```bash
npm run build
npm start
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Moiz Bhatti**
- GitHub: [@MoizBhatti333-code](https://github.com/MoizBhatti333-code)

## 🙏 Acknowledgments

- EfficientNet and MobileNetV2 models from TensorFlow/Keras
- Next.js team for the amazing framework
- Tailwind CSS for the utility-first CSS framework

---

Made with ❤️ using Next.js and TensorFlow
