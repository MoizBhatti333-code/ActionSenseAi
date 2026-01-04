"""
This script loads the trained model and generates captions for images.
"""

import os
import sys
import json
import numpy as np
import pickle
import math
from pathlib import Path

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, Bidirectional, LSTM, Concatenate, Layer
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Configuration
IMG_SIZE = 224
MAX_LENGTH = 40
ACTION_DIM = 1280
BEAM_INDEX = 3

# Paths (relative to this script)
BASE_DIR = Path(__file__).parent
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MODEL_WEIGHTS_PATH = BASE_DIR / "caption_trained_model.h5"


class BahdanauAttention(Layer):
    """Bahdanau Attention mechanism for caption generation"""
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, encoder_features, decoder_hidden):
        decoder_hidden_with_time_axis = tf.expand_dims(decoder_hidden, 1)
        score = tf.nn.tanh(self.W1(encoder_features) + self.W2(decoder_hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * encoder_features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, tf.squeeze(attention_weights, -1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class ImageCaptioningModel:
    """Image Captioning Model with Action Recognition"""
    
    def __init__(self):
        self.tokenizer = None
        self.vocab_size = None
        self.caption_model = None
        self.fe_model = None
        self.action_fe_model = None
        self.action_classifier = None
        self.num_patches = None
        self.feat_dim = None
        
    def load_models(self, verbose=True):
        """Load all required models and tokenizer"""
        # Load tokenizer
        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
        
        with open(TOKENIZER_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)
        
        self.vocab_size = len(self.tokenizer.word_index) + 1
        if verbose:
            print(f"Tokenizer loaded! Vocab size: {self.vocab_size}", file=sys.stderr)
        
        # Load feature extraction models
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        self.fe_model = Model(inputs=base_model.input, outputs=base_model.output)
        if verbose:
            print("EfficientNetB0 feature extractor loaded", file=sys.stderr)
        
        # Load action recognition models
        action_model_base = MobileNetV2(weights='imagenet', include_top=True, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        self.action_fe_model = Model(inputs=action_model_base.input, outputs=action_model_base.layers[-2].output)
        self.action_classifier = action_model_base
        if verbose:
            print("MobileNetV2 action recognition loaded", file=sys.stderr)
        
        # Get feature dimensions
        sample_img = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
        sample_feat = self.fe_model.predict(sample_img, verbose=0)[0]
        h, w, c = sample_feat.shape
        self.num_patches = h * w
        self.feat_dim = c
        if verbose:
            print(f"Feature dimensions: num_patches={self.num_patches}, feat_dim={self.feat_dim}", file=sys.stderr)
        
        # Build and load caption model
        self.caption_model = self._build_caption_model()
        
        if not MODEL_WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS_PATH}")
        
        self.caption_model.load_weights(str(MODEL_WEIGHTS_PATH))
        if verbose:
            print(f"Caption model weights loaded from {MODEL_WEIGHTS_PATH}", file=sys.stderr)
        
    def _build_caption_model(self):
        """Build caption model architecture"""
        # Input layers
        encoder_input = Input(shape=(self.num_patches, self.feat_dim), name='image_features')
        decoder_input = Input(shape=(MAX_LENGTH,), name='decoder_input')
        action_input = Input(shape=(ACTION_DIM,), name='action_features')
        
        # Embedding
        embedding_dim = 512
        embed = Embedding(input_dim=self.vocab_size, output_dim=embedding_dim, mask_zero=True, name='embed')(decoder_input)
        
        # Bidirectional LSTM
        bi_lstm = Bidirectional(LSTM(512, return_sequences=False, return_state=False), name='bilstm')(embed)
        
        # Attention
        attention_units = 512
        attention = BahdanauAttention(attention_units)
        context_vector, att_weights = attention(encoder_input, bi_lstm)
        
        # Action features processing
        action_dense = Dense(256, activation='relu', name='action_dense')(action_input)
        action_dropout = Dropout(0.3)(action_dense)
        
        # Decoder
        concat = Concatenate(axis=-1)([context_vector, bi_lstm, action_dropout])
        x = Dense(512, activation='relu')(concat)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.vocab_size, activation='softmax')(x)
        
        model = Model(inputs=[encoder_input, decoder_input, action_input], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        
        return model
    
    def extract_features(self, image_path):
        """Extract spatial features from image"""
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img)
        img = effnet_preprocess(img)
        img = np.expand_dims(img, axis=0)
        feat_map = self.fe_model.predict(img, verbose=0)
        feat_map = feat_map[0]
        h, w, c = feat_map.shape
        feat_reshaped = feat_map.reshape(-1, c)
        return feat_reshaped.astype(np.float32)
    
    def extract_action_features(self, image_path):
        """Extract action features from image"""
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img)
        img = mobilenet_preprocess(img)
        img = np.expand_dims(img, axis=0)
        action_feat = self.action_fe_model.predict(img, verbose=0)
        return action_feat[0].astype(np.float32)
    
    def predict_action(self, image_path):
        """Predict action class from image"""
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img)
        img = mobilenet_preprocess(img)
        img = np.expand_dims(img, axis=0)
        preds = self.action_classifier.predict(img, verbose=0)
        decoded = decode_predictions(preds, top=3)[0]
        
        # Return top 3 predictions
        results = []
        for class_id, class_name, confidence in decoded:
            results.append({
                "class": class_name.replace('_', ' ').title(),
                "confidence": float(confidence * 100)
            })
        return results
    
    def idx_to_word(self, integer):
        """Convert token index to word"""
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    
    def clean_caption(self, raw_caption):
        """Clean and format caption"""
        caption = raw_caption.replace('startseq', '').replace('endseq', '').strip()
        caption = caption.capitalize()
        if not caption.endswith('.'):
            caption = caption + '.'
        return caption
    
    def beam_search(self, feature, action_feat):
        """Generate caption using beam search"""
        start_token = self.tokenizer.word_index.get('startseq')
        end_token = self.tokenizer.word_index.get('endseq')
        
        if start_token is None or end_token is None:
            raise ValueError("startseq/endseq tokens missing in tokenizer")
        
        sequences = [[[start_token], 0.0]]
        
        while True:
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == end_token or len(seq) >= MAX_LENGTH:
                    all_candidates.append((seq, score))
                    continue
                
                sequence = pad_sequences([seq], maxlen=MAX_LENGTH, padding='post')
                yhat = self.caption_model.predict(
                    [feature[np.newaxis,...], sequence, action_feat[np.newaxis,...]],
                    verbose=0
                )[0]
                
                top_indices = np.argsort(yhat)[-BEAM_INDEX:]
                for idx in top_indices:
                    prob = yhat[idx]
                    if prob <= 0:
                        continue
                    candidate_seq = seq + [int(idx)]
                    candidate_score = score + math.log(prob + 1e-10)
                    all_candidates.append((candidate_seq, candidate_score))
            
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:BEAM_INDEX]
            
            if any(s[-1] == end_token for s, _ in sequences):
                break
            if len(sequences[0][0]) >= MAX_LENGTH:
                break
        
        best_seq = sequences[0][0]
        words = [self.idx_to_word(i) for i in best_seq]
        caption = " ".join([w for w in words if w is not None])
        return self.clean_caption(caption)
    
    def predict(self, image_path):
        """Generate caption and action predictions for an image"""
        # Extract features
        feature = self.extract_features(image_path)
        action_feat = self.extract_action_features(image_path)
        
        # Predict action
        action_results = self.predict_action(image_path)
        
        # Generate caption
        caption = self.beam_search(feature, action_feat)
        
        # Format response
        result = {
            "action_class": action_results[0]["class"],
            "confidence": f"{action_results[0]['confidence']:.1f}%",
            "annotations": [
                {"time": f"Prediction {i+1}", "text": f"{pred['class']} ({pred['confidence']:.1f}% confidence)"}
                for i, pred in enumerate(action_results[:3])
            ]
        }
        
        # Add caption as the main annotation
        result["annotations"].insert(0, {"time": "Caption", "text": caption})
        
        return result


# Global model instance (loaded once)
model_instance = None


def load_model_once():
    """Load model only once (singleton pattern)"""
    global model_instance
    if model_instance is None:
        model_instance = ImageCaptioningModel()
        model_instance.load_models(verbose=False)  # Suppress verbose output for API calls
    return model_instance


def main():
    """Main function for command-line usage"""
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python predict.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)
    
    try:
        # Load model
        model = load_model_once()
        
        # Make prediction
        result = model.predict(image_path)
        
        # Output as JSON
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
