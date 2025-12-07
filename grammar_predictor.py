import joblib
import numpy as np
import re
import os
import json

class GrammarErrorPredictor:
    def __init__(self, model_path='models'):
        try:
            self.model = joblib.load(f'{model_path}/best_model.joblib')
            self.scaler = joblib.load(f'{model_path}/scaler.joblib')
            self.label_encoder = joblib.load(f'{model_path}/label_encoder.joblib')
            
            base = os.path.abspath(model_path)
            self.model = joblib.load(os.path.join(base, 'best_model.joblib'))
            self.scaler = joblib.load(os.path.join(base, 'scaler.joblib'))
            self.label_encoder = joblib.load(os.path.join(base, 'label_encoder.joblib'))

            with open(os.path.join(base, 'metadata.json'), 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

            print("SUCCESS: Grammar Error Predictor loaded successfully!")
            print(f"INFO: Supports {len(self.metadata.get('target_classes', []))} error types")
            print(f"INFO: Best model: {self.metadata.get('best_model', 'unknown')}")
        except Exception as e:
            print(f"ERROR: Error loading model: {e}")
            raise
        # initialize spellchecker if available
        try:
            from spellchecker import SpellChecker
            self.spell = SpellChecker()
            self._SPELL_AVAILABLE = True
        except Exception:
            self.spell = None
            self._SPELL_AVAILABLE = False
    
    def extract_features(self, text):
        # Build features from the text
        features = {}
        text = str(text)

        features['char_count'] = len(text)

        # Extract words (lowercase)
        words = re.findall(r"\b\w+\b", text.lower())

        # If spellchecker available, create corrected words for feature calculation
        if self.spell and words:
            corrected_words = []
            for w in words:
                if len(w) <= 2:
                    corrected_words.append(w)
                    continue
                try:
                    if w not in self.spell:
                        suggestion = self.spell.correction(w)
                        corrected_words.append(suggestion if suggestion else w)
                    else:
                        corrected_words.append(w)
                except Exception:
                    corrected_words.append(w)
            words_for_features = corrected_words
        else:
            words_for_features = words

        features['word_count'] = len(words_for_features)

        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        features['sentence_count'] = max(1, sentence_endings)

        if features['word_count'] > 0:
            features['avg_word_length'] = features['char_count'] / features['word_count']
        else:
            features['avg_word_length'] = 0

        if features['sentence_count'] > 0:
            features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        else:
            features['avg_sentence_length'] = 0

        if len(text) > 0:
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
            features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text)
            features['punctuation_ratio'] = sum(1 for c in text if c in '.,;:!?\'"()[]{}') / len(text)
        else:
            features['uppercase_ratio'] = 0
            features['digit_ratio'] = 0
            features['punctuation_ratio'] = 0

        if words_for_features:
            features['unique_word_ratio'] = len(set(words_for_features)) / len(words_for_features)

            pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']
            articles = ['a', 'an', 'the']
            prepositions = ['in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of', 'from']

            features['has_pronoun'] = int(any(word in pronouns for word in words_for_features))
            features['has_article'] = int(any(word in articles for word in words_for_features))
            features['has_preposition'] = int(any(word in prepositions for word in words_for_features))

            text_lower = text.lower()
            features['has_verb_be'] = int(any(verb in text_lower for verb in [' am ', ' is ', ' are ', ' was ', ' were ', ' be ', ' been ']))
            features['has_modal'] = int(any(modal in text_lower for modal in [' can ', ' could ', ' will ', ' would ', ' shall ', ' should ', ' may ', ' might ', ' must ']))
            features['has_negative'] = int(' not ' in text_lower or "n't" in text_lower)
            features['has_apostrophe'] = int("'" in text)

            features['ends_with_period'] = int(text.strip().endswith('.'))
            features['starts_with_capital'] = int(text.strip() and text.strip()[0].isupper())
            features['has_comma'] = int(',' in text)
        else:
            features['unique_word_ratio'] = 0
            features['has_pronoun'] = 0
            features['has_article'] = 0
            features['has_preposition'] = 0
            features['has_verb_be'] = 0
            features['has_modal'] = 0
            features['has_negative'] = 0
            features['has_apostrophe'] = 0
            features['ends_with_period'] = 0
            features['starts_with_capital'] = 0
            features['has_comma'] = 0

        feature_array = []
        for feature_name in self.metadata['feature_names']:
            feature_array.append(features.get(feature_name, 0))

        return np.array(feature_array).reshape(1, -1)
    def predict(self, text):
        try:
            features = self.extract_features(text)
            features_scaled = self.scaler.transform(features)
            
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            error_type = self.label_encoder.inverse_transform([prediction])[0]
            
            top_indices = np.argsort(probability)[::-1][:3]
            top_predictions = []
            for idx in top_indices:
                top_predictions.append({
                    'error_type': self.label_encoder.classes_[idx],
                    'confidence': float(probability[idx])
                })
            # Heuristic overrides for known edge cases
            text_lower = str(text).lower()
            # Pronoun case heuristics
            pronoun_patterns = [r'between\s+you\s+and\s+i', r'\bme and \w+', r'\b\w+ and i\b']
            for pat in pronoun_patterns:
                if re.search(pat, text_lower):
                    if error_type != 'pronoun_case' and max(probability) < 0.7:
                        error_type = 'pronoun_case'
                        break

            # Irregular verb heuristic
            if re.search(r'\bhave\s+(saw|ate|swam|gone|run)\b', text_lower):
                if error_type != 'irregular_verb' and max(probability) < 0.7:
                    error_type = 'irregular_verb'

            return {
                'error_type': error_type,
                'confidence': float(max(probability)),
                'top_predictions': top_predictions
            }
        except Exception as e:
            return {
                'error': str(e),
                'error_type': 'unknown',
                'confidence': 0.0
            }

if __name__ == "__main__":
    predictor = GrammarErrorPredictor()
    
    test_sentences = [
        "He go to school every day.",
        "She don't like apples.",
        "Between you and I, it's wrong.",
        "I have saw that movie.",
        "A apple a day."
    ]
    
    print("\nTesting Grammar Error Predictor:")
    print("=" * 50)
    
    for sentence in test_sentences:
        result = predictor.predict(sentence)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            continue
            
        print(f"Sentence: {sentence}")
        print(f"Predicted Error: {result['error_type']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Top Predictions:")
        for pred in result['top_predictions']:
            print(f"  - {pred['error_type']}: {pred['confidence']:.2%}")
        print("-" * 50)
