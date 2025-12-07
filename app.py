import os
import re
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load models without numpy dependencies
def load_models_simple():
    """Load models without numpy dependency issues"""
    MODELS = {}
    MODELS_DIR = 'models'
    
    if not os.path.exists(MODELS_DIR):
        print("ℹ️  Models folder not found - using rule-based checking")
        return {}
    
    # Try to load with error handling
    try:
        import joblib
        import pickle
        model_files = [
            'best_model.joblib',
            'ensemble_model.joblib', 
            'gradient_boosting_model.joblib',
            'logistic_regression_model.joblib',
            'mlp_model.joblib',
            'random_forest_model.joblib'
        ]
        
        for filename in model_files:
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                try:
                    model_name = filename.replace('.joblib', '')
                    MODELS[model_name] = joblib.load(path)
                    print(f"✅ Loaded: {model_name}")
                except Exception as e:
                    # Provide detailed error and try a pickle fallback
                    print(f"⚠️  Skipped {filename}: {e}")
                    try:
                        with open(path, 'rb') as fh:
                            MODELS[model_name] = pickle.load(fh)
                        print(f"✅ Loaded with pickle fallback: {model_name}")
                    except Exception as e2:
                        print(f"  ❌ Fallback failed for {filename}: {e2}")
    except ImportError:
        print("ℹ️  Joblib not available - using rule-based checking")
    except Exception as e:
        print(f"ℹ️  Unexpected error while loading models: {e}")
    
    return MODELS

# Load models
MODELS = load_models_simple()

# Try to load ML predictor
try:
    from grammar_predictor import GrammarErrorPredictor
    predictor = GrammarErrorPredictor()
    print("✅ ML Grammar Predictor loaded")
except Exception as e:
    predictor = None
    print(f"ℹ️  ML Grammar Predictor not available: {e}")

# Try to load language tool
try:
    import language_tool_python
    tool = language_tool_python.LanguageTool('en-US')
    print("✅ LanguageTool loaded")
except:
    tool = None
    print("ℹ️  LanguageTool not available")

# Optional: Spell checker (pyspellchecker) for simple spelling corrections
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    print("✅ SpellChecker loaded")
except Exception:
    spell = None
    print("ℹ️  SpellChecker not available")

class GrammarChecker:
    def __init__(self):
        # Comprehensive correction database
        self.corrections = {
            # Spelling
            'wan': 'want',
            'thier': 'their',
            'teh': 'the',
            'adn': 'and',
            'yuor': 'your',
            'recieve': 'receive',
            'seperate': 'separate',
            'definately': 'definitely',
            'occured': 'occurred',
            'comming': 'coming',
            'truely': 'truly',
            'yersterday': 'yesterday',
            'tommorrow': 'tomorrow',
            'alot': 'a lot',

            # Modal verbs
            'could of': 'could have',
            'should of': 'should have',
            'would of': 'would have',
            'must of': 'must have',
            'may of': 'may have',
            'might of': 'might have',
            'can of': 'can have',
            'must to': 'must',

            'will can': 'can',
            'would can': 'could',
            'should can': 'could',
            'may can': 'may',
            'might can': 'might',
            'must can': 'must',
            'ought can': 'ought to',
            'dare can': 'dare to',
            'need can': 'need to',
            'used can': 'used to',
            'got can': 'got to',
            'has can': 'can',

            # Verb aspect (continuous/progressive)
            'am eat': 'am eating',
            'is eat': 'is eating',
            'are eat': 'are eating',
            'was eat': 'was eating',
            'were eat': 'were eating',
            'have eat': 'have eaten',
            'has eat': 'has eaten',
            'had eat': 'had eaten',
            'will eat': 'will be eating',
            'would eat': 'would be eating',
            'am walk': 'am walking',
            'is walk': 'is walking',
            'are walk': 'are walking',
            'was walk': 'was walking',
            'were walk': 'were walking',
            'have walk': 'have walked',
            'has walk': 'has walked',
            'had walk': 'had walked',
            'will walk': 'will be walking',
            'would walk': 'would be walking',
            'have been wait': 'have been waiting',
            'has been wait': 'has been waiting',
            'had been wait': 'had been waiting',

            # Comparatives and superlatives
            'more better': 'better',
            'more worse': 'worse',
            'more good': 'better',
            'more bad': 'worse',
            'most tallest': 'tallest',
            'most smallest': 'smallest',
            'most biggest': 'biggest',
            'most best': 'best',
            'most worst': 'worst',
            'gooder': 'better',
            'badder': 'worse',
            'more smarter': 'smarter',
            'more dumber': 'dumber',
            'most intelligentest': 'most intelligent',
            'most beautifulest': 'most beautiful',
            'more heavier': 'heavier',
            'more lighter': 'lighter',
            'more darker': 'darker',
            'more brighter': 'brighter',
            'more colder': 'colder',
            'more hotter': 'hotter',
            'more bigger': 'bigger',
            'more smaller': 'smaller',

            # Irregular verbs
            'swimmed': 'swam',
            'teached': 'taught',
            'bringed': 'brought',
            'fighted': 'fought',
            'thinked': 'thought',
            'mouses': 'mice',
            'mans': 'men',
            'childs': 'children',
            'gooses': 'geese',
            'tooths': 'teeth',
            'foots': 'feet',
            'sheeps': 'sheep',
            'fishes': 'fish',
            'cactuses': 'cacti',
            'crisises': 'crises',
            'phenomenons': 'phenomena',
            'criterias': 'criteria',
            'stimuli': 'stimuli',
            'nuclei': 'nuclei',
            'formulae': 'formulae',
            'alumni': 'alumni',
            'bacteria': 'bacteria',
            'curricula': 'curricula',
            'data': 'data',
            'media': 'media',
            'oxes': 'oxen',
            'deers': 'deer',
            'indices': 'indices',
            'matrices': 'matrices',
            'vertebrae': 'vertebrae',
            'libretti': 'libretti',
            'schemata': 'schemata',
            'anamneses': 'anamneses',
            'theses': 'theses',
            'analyses': 'analyses',
            'diagnoses': 'diagnoses',
            'synopses': 'synopses',
            'hypotheses': 'hypotheses',
            'parentheses': 'parentheses',
            'appendices': 'appendices',
            'drived': 'drove',
            'drawed': 'drew',
            'eated': 'ate',
            'finded': 'found',
            'flyed': 'flew',
            'forgetted': 'forgot',
            'getted': 'got',
            'gived': 'gave',
            'goed': 'went',
            'growed': 'grew',
            'knowed': 'knew',
            'layed': 'lay',
            'leaded': 'led',
            'leaved': 'left',
            'losed': 'lost',
            'maked': 'made',
            'meeted': 'met',
            'payed': 'paid',
            'rided': 'rode',
            'rised': 'rose',
            'runned': 'ran',
            'seed': 'saw',
            'selled': 'sold',
            'sended': 'sent',
            'shaked': 'shook',
            'shined': 'shone',
            'shuted': 'shut',
            'singed': 'sang',
            'sinked': 'sank',
            'sitted': 'sat',
            'sleeped': 'slept',
            'springed': 'sprang',
            'standed': 'stood',
            'stealed': 'stole',
            'sticked': 'stuck',
            'striked': 'struck',
            'swunged': 'swung',
            'throwed': 'threw',
            'understanded': 'understood',
            'waked': 'woke',
            'wear': 'wore',
            'weaved': 'wove',
            'winded': 'wound',
            'winned': 'won',
            'writed': 'wrote',

            # Reflexive pronouns
            'hisself': 'himself',
            'theirselves': 'themselves',
            'ourself': 'ourselves',
            'myself': 'I',  # Context-dependent, but common error
            'yourself': 'you',  # Context-dependent
            'Himself did': 'He did',
            'Myself will': 'I will',
            'Yourself should': 'You should',
            'Themself': 'Themselves',
            'Oneself': 'One',
            'Himself': 'He',  # Context-dependent
            'Herself': 'She',  # Context-dependent
            'Itself': 'It',  # Context-dependent

            # Verb forms (passive voice and others)
            'was threw': 'was thrown',
            'was wrote': 'was written',
            'was broke': 'was broken',
            'was ate': 'was eaten',
            'was drank': 'was drunk',
            'was drove': 'was driven',
            'was rode': 'was ridden',
            'was spoke': 'was spoken',
            'was sung': 'was sung',
            'was broke': 'was broken',
            'was stole': 'was stolen',
            'was tore': 'was torn',
            'was wore': 'was worn',
            'were threw': 'were thrown',
            'were wrote': 'were written',
            'were broke': 'were broken',
            'were ate': 'were eaten',
            'were drank': 'were drunk',
            'were drove': 'were driven',
            'were rode': 'were ridden',
            'were spoke': 'were spoken',
            'were sung': 'were sung',
            'were broke': 'were broken',
            'were stole': 'were stolen',
            'were tore': 'were torn',
            'were wore': 'were worn',
            'is threw': 'is thrown',
            'is wrote': 'is written',
            'is broke': 'is broken',
            'is ate': 'is eaten',
            'is drank': 'is drunk',
            'is drove': 'is driven',
            'is rode': 'is ridden',
            'is spoke': 'is spoken',
            'is sung': 'is sung',
            'is broke': 'is broken',
            'is stole': 'is stolen',
            'is tore': 'is torn',
            'is wore': 'is worn',
            'are threw': 'are thrown',
            'are wrote': 'are written',
            'are broke': 'are broken',
            'are ate': 'are eaten',
            'are drank': 'are drunk',
            'are drove': 'are driven',
            'are rode': 'are ridden',
            'are spoke': 'are spoken',
            'are sung': 'are sung',
            'are broke': 'are broken',
            'are stole': 'are stolen',
            'are tore': 'are torn',
            'are wore': 'are worn',
            'am threw': 'am thrown',
            'am wrote': 'am written',
            'am broke': 'am broken',
            'am ate': 'am eaten',
            'am drank': 'am drunk',
            'am drove': 'am driven',
            'am rode': 'am ridden',
            'am spoke': 'am spoken',
            'am sung': 'am sung',
            'am broke': 'am broken',
            'am stole': 'am stolen',
            'am tore': 'am torn',
            'am wore': 'am worn',

            # Contractions
            'its': "it's",
            'your': "you're",
            'theyre': "they're",
            'theres': "there's",
            'wheres': "where's",
            'whos': "who's",

            # Subject-verb agreement
            'he go': 'he goes',
            'she go': 'she goes',
            'it go': 'it goes',
            'he have': 'he has',
            'she have': 'she has',
            'it have': 'it has',
            'he do': 'he does',
            'she do': 'she does',
            'it do': 'it does',
            'they goes': 'they go',
            'they has': 'they have',
            'they does': 'they do',

            # Verb to be
            'i is': 'I am',
            'you is': 'you are',
            'we is': 'we are',
            'they is': 'they are',
            'i are': 'I am',
            'you am': 'you are',
            'we am': 'we are',
            'they am': 'they are',
            'he are': 'he is',
            'she are': 'she is',
            'it are': 'it is',

            # Articles
            'a apple': 'an apple',
            'a egg': 'an egg',
            'a hour': 'an hour',
            'a honest': 'an honest',
            'an book': 'a book',
            'an car': 'a car',
            'an university': 'a university',
            'an user': 'a user',

            # Prepositions
            'interested about': 'interested in',
            'depends of': 'depends on',
            'listen music': 'listen to music',
            'arrive to': 'arrive at',
            'complain for': 'complain about',
            'married with': 'married to',
            'good in': 'good at'
        }
        
        # Special patterns with custom corrections
        self.special_patterns = [
            # Tense: will + yesterday -> did
            (r'\b(will)\s+(.*?)\s+(yesterday)\b', 
             lambda m: f"did {m.group(2)} yesterday",
             "Tense Error: Use past tense with 'yesterday'"),
            
            # Tense: did + tomorrow -> will
            (r'\b(did)\s+(.*?)\s+(tomorrow)\b',
             lambda m: f"will {m.group(2)} tomorrow",
             "Tense Error: Use future tense with 'tomorrow'"),
            
            # Missing 'to' before verb
            (r'\b(want|need|try|like)\s+(\w+)\b(?!\s+to\b)',
             lambda m: f"{m.group(1)} to {m.group(2)}",
             "Grammar: Add 'to' before verb"),
            
            # Missing 's' for third person
            (r'\b(he|she|it)\s+(\w+[^s])\b(?!\s+ing\b)',
             lambda m: f"{m.group(1)} {m.group(2)}s",
             "Grammar: Add 's' for third person singular")
        ]
        
        print("✅ Grammar checker initialized")
    
    def find_errors(self, text):
        """Find all grammar errors in text"""
        errors = []
        # Keep original text for accurate offsets
        original_text = text
        # Work on a modifiable copy for rule detection; record spelling errors separately
        text_lower = text.lower()

        # Early spelling pass: correct obvious misspellings so rules (tense etc.) see corrected tokens
        if spell:
            corrected_tokens = []
            word_matches = list(re.finditer(r"\b[a-zA-Z']+\b", text))
            last = 0
            corrected_text = text
            spelling_errors = []

            # Build corrected text by replacing misspelled tokens with suggestions
            for m in reversed(word_matches):
                w = m.group(0)
                wl = w.lower()
                # Skip short/common words
                if len(wl) <= 2:
                    continue
                common = {'a','an','the','i','you','he','she','it','we','they','me','him','her','us','them','to','of','in','on','at','for','with','by','and','or','but','yesterday','tomorrow'}
                if wl in common:
                    continue
                try:
                    if wl not in spell:
                        suggestion = spell.correction(wl)
                        if suggestion and suggestion != wl:
                            # Preserve capitalization
                            if w[0].isupper():
                                suggestion = suggestion.capitalize()
                            # Replace in corrected_text using slice positions
                            start, end = m.start(), m.end()
                            corrected_text = corrected_text[:start] + suggestion + corrected_text[end:]
                            spelling_errors.append({
                                'text': w,
                                'correction': suggestion,
                                'type': 'Spelling',
                                'severity': 'medium',
                                'start': start,
                                'end': end
                            })
                except Exception:
                    pass

            # If we found spelling suggestions, record them and use corrected text for further rules
            if spelling_errors:
                errors.extend(spelling_errors)
                text = corrected_text
                text_lower = text.lower()
        
        # 1. Check for tense inconsistencies
        # Past tense with time expressions like "yesterday"
        time_expressions_past = ['yesterday', 'last week', 'last month', 'last year', 'last night', 'ago']
        verb_list = {'tell', 'say', 'go', 'come', 'see', 'do', 'make', 'take', 'get', 'give', 'know', 'think', 'want', 'need', 'feel', 'look', 'ask', 'work', 'call', 'try', 'use', 'find', 'live', 'leave', 'put', 'mean', 'keep', 'let', 'begin', 'help', 'show', 'hear', 'play', 'run', 'move', 'write', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull'}

        print(f"DEBUG: Text: '{text}'")
        print(f"DEBUG: Time expressions found: {[expr for expr in time_expressions_past if expr in text_lower]}")

        if any(expr in text_lower for expr in time_expressions_past):
            print("DEBUG: Entering tense check")
            # Find subject-verb pairs that should be past tense
            subject_verb_pattern = r'\b(i|you|we|they|he|she|it)\s+(\w+)\b'
            for match in re.finditer(subject_verb_pattern, text, re.IGNORECASE):
                subject = match.group(1)
                verb = match.group(2).lower()
                print(f"DEBUG: Subject-verb match: '{subject} {verb}' at {match.start()}-{match.end()}")
                if verb in verb_list:
                    print(f"DEBUG: Verb '{verb}' is in verb_list")
                    # Check if this verb is near a past time expression
                    verb_start = match.start()
                    verb_end = match.end()
                    # Look for time expressions within reasonable distance
                    context_start = max(0, verb_start - 50)
                    context_end = min(len(text), verb_end + 50)
                    context = text[context_start:context_end].lower()

                    if any(expr in context for expr in time_expressions_past):
                        print(f"DEBUG: Time expression found in context: '{context}'")
                        wrong_text = text[verb_start:verb_end]
                        corrected = f"{subject} {self._get_past_tense(verb)}"
                        print(f"DEBUG: Adding tense error: '{wrong_text}' -> '{corrected}'")
                        errors.append({
                            'text': wrong_text,
                            'correction': corrected,
                            'type': 'Tense Error',
                            'severity': 'high',
                            'start': verb_start,
                            'end': verb_end
                        })

            # Find verb-object pairs that should be past tense (only for modal cases to avoid overlap)
            verb_object_pattern = r'\b((will|would|can|could|should|may|might)\s+)?(\w+)\s+(him|her|them|it|us|me)\b'
            for match in re.finditer(verb_object_pattern, text, re.IGNORECASE):
                modal = match.group(2).lower() if match.group(2) else None
                verb = match.group(3).lower()
                obj = match.group(4)
                print(f"DEBUG: Verb-object match: '{modal} {verb} {obj}' at {match.start()}-{match.end()}")
                if verb in verb_list and modal is not None:  # Only apply for modal cases
                    print(f"DEBUG: Verb '{verb}' is in verb_list and modal '{modal}' present")
                    # Check if this verb is near a past time expression
                    verb_start = match.start()
                    verb_end = match.end()
                    # Look for time expressions within reasonable distance
                    context_start = max(0, verb_start - 50)
                    context_end = min(len(text), verb_end + 50)
                    context = text[context_start:context_end].lower()

                    if any(expr in context for expr in time_expressions_past):
                        print(f"DEBUG: Time expression found in context: '{context}'")
                        wrong_text = text[verb_start:verb_end]
                        corrected = f"{self._get_past_tense(verb)} {obj}"
                        print(f"DEBUG: Adding tense error: '{wrong_text}' -> '{corrected}'")
                        errors.append({
                            'text': wrong_text,
                            'correction': corrected,
                            'type': 'Tense Error',
                            'severity': 'high',
                            'start': verb_start,
                            'end': verb_end
                        })

        # Future tense with "tomorrow"
        if 'tomorrow' in text_lower:
            pattern = r'\b(did|was)\s+(.*?)\s+(tomorrow)\b'
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                start, end = match.span()
                wrong_text = text[start:end]
                replacement = f"will {match.group(2)} tomorrow"
                errors.append({
                    'text': wrong_text,
                    'correction': replacement,
                    'type': 'Tense Error',
                    'severity': 'high',
                    'start': start,
                    'end': end
                })

        # Check for infinitive after "to" (should not be past participle)
        infinitive_patterns = [
            r'\bto\s+(done|gone|seen|eaten|drunk|run|swum|sung|driven|spoken|broken|written|taken|given|known|thought|made|said|come|become|been|had)\b'
        ]

        for pattern in infinitive_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                wrong_text = text[start:end]
                verb = match.group(1)
                # Convert past participle back to infinitive
                infinitive = self._past_participle_to_infinitive(verb.lower())
                corrected = f"to {infinitive}"
                errors.append({
                    'text': wrong_text,
                    'correction': corrected,
                    'type': 'Verb Form',
                    'severity': 'high',
                    'start': start,
                    'end': end
                })
        
        # 2. Check for perfect tense verb form errors
        # Present perfect: have/has + past participle (not simple past)
        perfect_tense_patterns = [
            (r'\bhave\s+(saw|ate|went|ran|swam|drank|drove|sang|spoke|broke|wrote)\b',
             lambda m: f"have {self._past_to_participle(m.group(1))}"),
            (r'\bhas\s+(saw|ate|went|ran|swam|drank|drove|sang|spoke|broke|wrote)\b',
             lambda m: f"has {self._past_to_participle(m.group(1))}")
        ]

        for pattern, correction_func in perfect_tense_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                wrong_text = text[start:end]
                corrected = correction_func(match)
                errors.append({
                    'text': wrong_text,
                    'correction': corrected,
                    'type': 'Verb Form',
                    'severity': 'high',
                    'start': start,
                    'end': end
                })

        # 3. Check common errors from dictionary
        # Dictionary-based corrections (use case-insensitive matching on the current text)
        for wrong, right in self.corrections.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                start, end = match.span()
                original = text[start:end]

                # Preserve capitalization
                if original[0].isupper():
                    corrected = right[0].upper() + right[1:]
                else:
                    corrected = right

                # Determine error type
                if len(wrong.split()) == 1:
                    error_type = 'Spelling'
                elif 'of' in wrong or 'to' in wrong or 'in' in wrong:
                    error_type = 'Grammar'
                else:
                    error_type = 'Usage Error'

                errors.append({
                    'text': original,
                    'correction': corrected,
                    'type': error_type,
                    'severity': 'medium',
                    'start': start,
                    'end': end
                })

        # 4. Rule-based detection for modal verbs
        modal_patterns = [
            (r'\b(can|could|will|would|shall|should|may|might|must)\s+(\w+s)\b', 'remove_s'),
            (r'\b(can|could|will|would|shall|should|may|might|must)\s+(of)\b', 'of_to_have'),
            (r'\b(can|could|will|would|shall|should|may|might|must)\s+(to)\s+(\w+)\b', 'remove_to')
        ]
        for pattern, correction_type in modal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                wrong_text = text[start:end]
                if correction_type == 'remove_s':
                    modal, verb_with_s = match.groups()[:2]
                    corrected = f"{modal} {verb_with_s[:-1]}"
                elif correction_type == 'of_to_have':
                    modal = match.group(1)
                    corrected = f"{modal} have"
                elif correction_type == 'remove_to':
                    modal, verb = match.groups()[0], match.groups()[2]
                    corrected = f"{modal} {verb}"
                errors.append({
                    'text': wrong_text,
                    'correction': corrected,
                    'type': 'Modal Verb',
                    'severity': 'high',
                    'start': start,
                    'end': end
                })

        # 5. Rule-based detection for subjunctive mood
        subjunctive_patterns = [
            (r'\b(if\s+i\s+)was\b', r'\1were'),
            (r'\b(if\s+you\s+)was\b', r'\1were'),
            (r'\b(if\s+we\s+)was\b', r'\1were'),
            (r'\b(if\s+they\s+)was\b', r'\1were'),
            (r'\b(if\s+he\s+)was\b', r'\1were'),
            (r'\b(if\s+she\s+)was\b', r'\1were'),
            (r'\b(if\s+it\s+)was\b', r'\1were'),
            (r'\b(i\s+suggest\s+that\s+he\s+)goes\b', r'\1go'),
            (r'\b(i\s+suggest\s+that\s+she\s+)goes\b', r'\1go'),
            (r'\b(i\s+suggest\s+that\s+it\s+)goes\b', r'\1go'),
            (r'\b(i\s+suggest\s+that\s+we\s+)go\b', r'\1go'),
            (r'\b(i\s+suggest\s+that\s+they\s+)go\b', r'\1go'),
            (r'\b(it\s+is\s+important\s+that\s+he\s+)is\b', r'\1be'),
            (r'\b(it\s+is\s+important\s+that\s+she\s+)is\b', r'\1be'),
            (r'\b(it\s+is\s+important\s+that\s+it\s+)is\b', r'\1be'),
            (r'\b(it\s+is\s+important\s+that\s+we\s+)are\b', r'\1be'),
            (r'\b(it\s+is\s+important\s+that\s+they\s+)are\b', r'\1be')
        ]
        for pattern, replacement in subjunctive_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                wrong_text = text[start:end]
                corrected = re.sub(pattern, replacement, wrong_text, flags=re.IGNORECASE)
                errors.append({
                    'text': wrong_text,
                    'correction': corrected,
                    'type': 'Subjunctive Mood',
                    'severity': 'medium',
                    'start': start,
                    'end': end
                })

        # 6. Spelling check (using pyspellchecker) - run once and add offsets
        if spell:
            word_matches = list(re.finditer(r"\b[a-zA-Z']+\b", text))
            existing_texts = {e['text'].lower() for e in errors if 'text' in e}

            for m in word_matches:
                w = m.group(0)
                wl = w.lower()
                start, end = m.start(), m.end()

                # Skip if already detected or short tokens or common words
                if wl in existing_texts or len(wl) <= 2:
                    continue

                common = {'a','an','the','i','you','he','she','it','we','they','me','him','her','us','them','to','of','in','on','at','for','with','by','and','or','but','yesterday','tomorrow'}
                if wl in common:
                    continue

                try:
                    if wl not in spell:
                        suggestion = spell.correction(wl)
                        if suggestion and suggestion != wl:
                            # Preserve capitalization
                            if w[0].isupper():
                                suggestion = suggestion.capitalize()

                            errors.append({
                                'text': w,
                                'correction': suggestion,
                                'type': 'Spelling',
                                'severity': 'medium',
                                'start': start,
                                'end': end
                            })
                except Exception:
                    pass

        # Deduplicate errors by (start,end,type,correction) keeping first occurrence
        seen = set()
        unique_errors = []
        for e in errors:
            key = (e.get('start'), e.get('end'), e.get('type'), str(e.get('correction','')).lower())
            if key in seen:
                continue
            seen.add(key)
            unique_errors.append(e)

        errors = unique_errors
        
        # 3. Check for missing punctuation
        if text.strip() and text.strip()[-1] not in '.!?':
            errors.append({
                'text': 'Missing ending punctuation',
                'correction': 'Add . ! or ?',
                'type': 'Punctuation',
                'severity': 'low'
            })
        
        # 4. Check capitalization
        if text and text[0].islower():
            errors.append({
                'text': 'Sentence starts with lowercase',
                'correction': 'Capitalize first letter',
                'type': 'Capitalization',
                'severity': 'low'
            })
        
        # 5. Use ML predictor for advanced error detection
        if predictor:
            try:
                prediction = predictor.predict(text)
                if prediction and 'error_type' in prediction and prediction['confidence'] > 0.6:
                    error_type = prediction['error_type']

                    # Map ML error types to our error types and provide corrections
                    if error_type == 'subject_verb_agreement':
                        # Look for subject-verb mismatches
                        words = re.findall(r'\b\w+\b', text)
                        for i, word in enumerate(words):
                            if word.lower() in ['he', 'she', 'it'] and i + 1 < len(words):
                                next_word = words[i + 1].lower()
                                if next_word in ['go', 'have', 'do', 'be']:
                                    verb_base = next_word
                                    if verb_base == 'be':
                                        verb_base = 'is'
                                    elif verb_base in ['go', 'do']:
                                        verb_base = verb_base + 'es'
                                    elif verb_base == 'have':
                                        verb_base = 'has'
                                    else:
                                        verb_base = verb_base + 's'

                                    start = text.lower().find(f"{word} {words[i+1]}")
                                    if start != -1:
                                        end = start + len(f"{word} {words[i+1]}")
                                        errors.append({
                                            'text': text[start:end],
                                            'correction': f"{word} {verb_base}",
                                            'type': 'Subject-Verb Agreement',
                                            'severity': 'high',
                                            'start': start,
                                            'end': end
                                        })

                    elif error_type == 'tense_error':
                        # Enhanced tense detection
                        if re.search(r'\b(will|would|can|could|should|may|might)\s+(\w+)\s+(yesterday|last\s+\w+)\b', text, re.IGNORECASE):
                            match = re.search(r'\b(will|would|can|could|should|may|might)\s+(\w+)\s+(yesterday|last\s+\w+)\b', text, re.IGNORECASE)
                            if match:
                                modal, verb, time = match.groups()
                                past_verb = self._get_past_tense(verb)
                                start, end = match.span()
                                errors.append({
                                    'text': text[start:end],
                                    'correction': f"{modal} have {past_verb} {time}",
                                    'type': 'Tense Error',
                                    'severity': 'high',
                                    'start': start,
                                    'end': end
                                })

                    elif error_type == 'verb_form':
                        # Check for incorrect verb forms in perfect tenses
                        irregular_verbs = {
                            'go': 'gone', 'be': 'been', 'have': 'had', 'do': 'done',
                            'see': 'seen', 'eat': 'eaten', 'drink': 'drunk', 'run': 'run',
                            'swim': 'swum', 'sing': 'sung', 'drive': 'driven',
                            'write': 'written', 'speak': 'spoken', 'break': 'broken'
                        }

                        # Check for "have + past" that should be "have + past participle"
                        for base, participle in irregular_verbs.items():
                            past_form = self._get_past_tense(base)
                            if past_form != participle:  # Only if they differ
                                pattern = r'\bhave\s+' + past_form + r'\b'
                                if re.search(pattern, text, re.IGNORECASE):
                                    match = re.search(pattern, text, re.IGNORECASE)
                                    start, end = match.span()
                                    errors.append({
                                        'text': text[start:end],
                                        'correction': f"have {participle}",
                                        'type': 'Verb Form',
                                        'severity': 'high',
                                        'start': start,
                                        'end': end
                                    })

                        # Also check for present perfect with wrong auxiliary
                        present_perfect_patterns = [
                            (r'\b(he|she|it)\s+have\s+(\w+)\b', r'\1 has \2'),
                            (r'\b(i|you|we|they)\s+has\s+(\w+)\b', r'\1 have \2')
                        ]
                        for pattern, replacement in present_perfect_patterns:
                            for match in re.finditer(pattern, text, re.IGNORECASE):
                                start, end = match.span()
                                corrected = re.sub(pattern, replacement, text[start:end], flags=re.IGNORECASE)
                                errors.append({
                                    'text': text[start:end],
                                    'correction': corrected,
                                    'type': 'Subject-Verb Agreement',
                                    'severity': 'high',
                                    'start': start,
                                    'end': end
                                })

                    elif error_type == 'passive_voice':
                        # Detect passive voice patterns and suggest active voice
                        passive_patterns = [
                            (r'\b(is|are|was|were|be|been|being)\s+(\w+ed|\w+en)\s+(by\s+\w+)\b', 'active_voice'),
                            (r'\b(has|have|had)\s+been\s+(\w+ed|\w+en)\s+(by\s+\w+)\b', 'active_voice')
                        ]
                        for pattern, _ in passive_patterns:
                            for match in re.finditer(pattern, text, re.IGNORECASE):
                                start, end = match.span()
                                passive_phrase = text[start:end]
                                # Simple active voice suggestion
                                errors.append({
                                    'text': passive_phrase,
                                    'correction': 'Consider using active voice',
                                    'type': 'Passive Voice',
                                    'severity': 'medium',
                                    'start': start,
                                    'end': end
                                })

                    elif error_type == 'modal_verb':
                        # Detect modal verb errors
                        modal_errors = [
                            (r'\b(can|could|will|would|shall|should|may|might|must)\s+(\w+s)\b', lambda m: f"{m.group(1)} {m.group(2)[:-1]}"),  # Remove 's' after modal
                            (r'\b(can|could|will|would|shall|should|may|might|must)\s+(of)\b', lambda m: f"{m.group(1)} have"),  # 'of' to 'have'
                            (r'\b(can|could|will|would|shall|should|may|might|must)\s+(to)\s+(\w+)\b', lambda m: f"{m.group(1)} {m.group(3)}")  # Remove 'to'
                        ]
                        for pattern, correction_func in modal_errors:
                            for match in re.finditer(pattern, text, re.IGNORECASE):
                                start, end = match.span()
                                wrong_text = text[start:end]
                                corrected = correction_func(match)
                                errors.append({
                                    'text': wrong_text,
                                    'correction': corrected,
                                    'type': 'Modal Verb',
                                    'severity': 'high',
                                    'start': start,
                                    'end': end
                                })

                    elif error_type == 'subjunctive_mood':
                        # Detect subjunctive mood errors
                        subjunctive_patterns = [
                            (r'\b(if\s+i\s+)was\b', r'\1were'),
                            (r'\b(if\s+you\s+)was\b', r'\1were'),
                            (r'\b(if\s+we\s+)was\b', r'\1were'),
                            (r'\b(if\s+they\s+)was\b', r'\1were'),
                            (r'\b(if\s+he\s+)was\b', r'\1were'),
                            (r'\b(if\s+she\s+)was\b', r'\1were'),
                            (r'\b(if\s+it\s+)was\b', r'\1were'),
                            (r'\b(i\s+suggest\s+that\s+he\s+)goes\b', r'\1go'),
                            (r'\b(i\s+suggest\s+that\s+she\s+)goes\b', r'\1go'),
                            (r'\b(i\s+suggest\s+that\s+it\s+)goes\b', r'\1go'),
                            (r'\b(i\s+suggest\s+that\s+we\s+)go\b', r'\1go'),
                            (r'\b(i\s+suggest\s+that\s+they\s+)go\b', r'\1go'),
                            (r'\b(it\s+is\s+important\s+that\s+he\s+)is\b', r'\1be'),
                            (r'\b(it\s+is\s+important\s+that\s+she\s+)is\b', r'\1be'),
                            (r'\b(it\s+is\s+important\s+that\s+it\s+)is\b', r'\1be'),
                            (r'\b(it\s+is\s+important\s+that\s+we\s+)are\b', r'\1be'),
                            (r'\b(it\s+is\s+important\s+that\s+they\s+)are\b', r'\1be')
                        ]
                        for pattern, replacement in subjunctive_patterns:
                            for match in re.finditer(pattern, text, re.IGNORECASE):
                                start, end = match.span()
                                wrong_text = text[start:end]
                                corrected = re.sub(pattern, replacement, wrong_text, flags=re.IGNORECASE)
                                errors.append({
                                    'text': wrong_text,
                                    'correction': corrected,
                                    'type': 'Subjunctive Mood',
                                    'severity': 'medium',
                                    'start': start,
                                    'end': end
                                })

            except Exception as e:
                print(f"ML predictor error: {e}")

        # 6. Use LanguageTool if available
        if tool:
            try:
                matches = tool.check(text)
                for match in matches[:5]:  # Limit for speed
                    if match.replacements:
                        error_text = text[match.offset:match.offset + match.errorLength]
                        errors.append({
                            'text': error_text,
                            'correction': match.replacements[0],
                            'type': 'Grammar',
                            'severity': 'medium'
                        })
            except:
                pass

        return errors

    def _get_past_tense(self, verb):
        """Simple past tense converter"""
        verb = verb.lower()
        irregular_past = {
            'go': 'went', 'be': 'was/were', 'have': 'had', 'do': 'did',
            'see': 'saw', 'eat': 'ate', 'drink': 'drank', 'run': 'ran',
            'swim': 'swam', 'sing': 'sang', 'drive': 'drove',
            'tell': 'told', 'say': 'said', 'make': 'made', 'take': 'took',
            'get': 'got', 'give': 'gave', 'know': 'knew', 'think': 'thought',
            'want': 'wanted', 'need': 'needed', 'feel': 'felt', 'look': 'looked',
            'ask': 'asked', 'work': 'worked', 'call': 'called', 'try': 'tried',
            'use': 'used', 'find': 'found', 'live': 'lived', 'leave': 'left',
            'put': 'put', 'mean': 'meant', 'keep': 'kept', 'let': 'let',
            'begin': 'began', 'help': 'helped', 'show': 'showed', 'hear': 'heard',
            'play': 'played', 'run': 'ran', 'move': 'moved', 'write': 'wrote',
            'sit': 'sat', 'stand': 'stood', 'lose': 'lost', 'pay': 'paid',
            'meet': 'met', 'include': 'included', 'continue': 'continued', 'set': 'set',
            'learn': 'learned', 'change': 'changed', 'lead': 'led', 'understand': 'understood',
            'watch': 'watched', 'follow': 'followed', 'stop': 'stopped', 'create': 'created',
            'speak': 'spoke', 'read': 'read', 'allow': 'allowed', 'add': 'added',
            'spend': 'spent', 'grow': 'grew', 'open': 'opened', 'walk': 'walked',
            'win': 'won', 'offer': 'offered', 'remember': 'remembered', 'love': 'loved',
            'consider': 'considered', 'appear': 'appeared', 'buy': 'bought', 'wait': 'waited',
            'serve': 'served', 'die': 'died', 'send': 'sent', 'expect': 'expected',
            'build': 'built', 'stay': 'stayed', 'fall': 'fell', 'cut': 'cut',
            'reach': 'reached', 'kill': 'killed', 'remain': 'remained', 'suggest': 'suggested',
            'raise': 'raised', 'pass': 'passed', 'sell': 'sold', 'require': 'required',
            'report': 'reported', 'decide': 'decided', 'pull': 'pulled'
        }
        if verb in irregular_past:
            return irregular_past[verb]
        elif verb.endswith('e'):
            return verb + 'd'
        elif verb.endswith(('y', 'p', 'k', 'ch', 'sh', 'x', 'z')):
            return verb[:-1] + 'ied' if verb.endswith('y') else verb + 'ed'
        else:
            return verb + 'ed'

    def _get_past_participle(self, verb):
        """Simple past participle converter"""
        verb = verb.lower()
        irregular_participle = {
            'go': 'gone', 'be': 'been', 'have': 'had', 'do': 'done',
            'see': 'seen', 'eat': 'eaten', 'drink': 'drunk', 'run': 'run',
            'swim': 'swum', 'sing': 'sung', 'drive': 'driven',
            'tell': 'told', 'say': 'said', 'make': 'made', 'take': 'taken',
            'get': 'gotten', 'give': 'given', 'know': 'known', 'think': 'thought',
            'want': 'wanted', 'need': 'needed', 'feel': 'felt', 'look': 'looked',
            'ask': 'asked', 'work': 'worked', 'call': 'called', 'try': 'tried',
            'use': 'used', 'find': 'found', 'live': 'lived', 'leave': 'left',
            'put': 'put', 'mean': 'meant', 'keep': 'kept', 'let': 'let',
            'begin': 'begun', 'help': 'helped', 'show': 'shown', 'hear': 'heard',
            'play': 'played', 'run': 'run', 'move': 'moved', 'write': 'written',
            'sit': 'sat', 'stand': 'stood', 'lose': 'lost', 'pay': 'paid',
            'meet': 'met', 'include': 'included', 'continue': 'continued', 'set': 'set',
            'learn': 'learned', 'change': 'changed', 'lead': 'led', 'understand': 'understood',
            'watch': 'watched', 'follow': 'followed', 'stop': 'stopped', 'create': 'created',
            'speak': 'spoken', 'read': 'read', 'allow': 'allowed', 'add': 'added',
            'spend': 'spent', 'grow': 'grown', 'open': 'opened', 'walk': 'walked',
            'win': 'won', 'offer': 'offered', 'remember': 'remembered', 'love': 'loved',
            'consider': 'considered', 'appear': 'appeared', 'buy': 'bought', 'wait': 'waited',
            'serve': 'served', 'die': 'died', 'send': 'sent', 'expect': 'expected',
            'build': 'built', 'stay': 'stayed', 'fall': 'fallen', 'cut': 'cut',
            'reach': 'reached', 'kill': 'killed', 'remain': 'remained', 'suggest': 'suggested',
            'raise': 'raised', 'pass': 'passed', 'sell': 'sold', 'require': 'required',
            'report': 'reported', 'decide': 'decided', 'pull': 'pulled'
        }
        if verb in irregular_participle:
            return irregular_participle[verb]
        elif verb.endswith('e'):
            return verb + 'd'
        elif verb.endswith(('y', 'p', 'k', 'ch', 'sh', 'x', 'z')):
            return verb[:-1] + 'ied' if verb.endswith('y') else verb + 'ed'
        else:
            return verb + 'ed'

    def _past_to_participle(self, past_verb):
        """Convert past tense to past participle"""
        past_verb = past_verb.lower()
        # Map common past forms to their participles
        past_to_participle = {
            'saw': 'seen', 'ate': 'eaten', 'went': 'gone', 'ran': 'run',
            'swam': 'swum', 'drank': 'drunk', 'drove': 'driven', 'sang': 'sung',
            'spoke': 'spoken', 'broke': 'broken', 'wrote': 'written'
        }
        return past_to_participle.get(past_verb, past_verb)  # Return original if not found

    def _past_participle_to_infinitive(self, participle):
        """Convert past participle back to infinitive"""
        participle = participle.lower()
        # Map common participles back to infinitives
        participle_to_infinitive = {
            'gone': 'go', 'been': 'be', 'had': 'have', 'done': 'do',
            'seen': 'see', 'eaten': 'eat', 'drunk': 'drink', 'run': 'run',
            'swum': 'swim', 'sung': 'sing', 'driven': 'drive',
            'written': 'write', 'spoken': 'speak', 'broken': 'break',
            'taken': 'take', 'given': 'give', 'known': 'know', 'thought': 'think',
            'made': 'make', 'said': 'say', 'come': 'come', 'become': 'become'
        }
        return participle_to_infinitive.get(participle, participle)  # Return original if not found
    
    def correct_text(self, text, errors):
        """Apply corrections to text"""
        if not errors:
            return text
        corrected = text

        # Apply corrections using offsets when available to avoid accidental multiple replacements
        # Collect edits that have start/end
        edits = [e for e in errors if 'start' in e and 'end' in e and e.get('correction')]

        if edits:
            # Sort by start descending so replacements don't shift earlier indexes
            edits_sorted = sorted(edits, key=lambda x: x['start'], reverse=True)
            for e in edits_sorted:
                s, t = e['start'], e['end']
                replacement = e.get('correction', '')
                # Safety: ensure indices within current string bounds
                if 0 <= s < len(corrected) and 0 <= t <= len(corrected) and s < t:
                    corrected = corrected[:s] + replacement + corrected[t:]

        # For any remaining corrections without offsets (messages like punctuation or capitalization), apply after
        for error in errors:
            wrong = error.get('text', '')
            right = error.get('correction', '')
            if not right or not wrong:
                continue
            # Skip ones already applied via offsets
            if 'start' in error and 'end' in error:
                continue
            if wrong.startswith('Missing'):
                continue
            # Simple replace as a fallback
            corrected = corrected.replace(wrong, right)
        
        # Ensure ending punctuation
        if corrected.strip() and corrected.strip()[-1] not in '.!?':
            corrected = corrected.strip() + '.'
        
        # Ensure capitalization
        if corrected and corrected[0].islower():
            corrected = corrected[0].upper() + corrected[1:]
        
        # Fix common patterns
        corrected = corrected.replace(' i ', ' I ')
        corrected = corrected.replace(' i,', ' I,')
        corrected = corrected.replace(' i.', ' I.')
        
        return corrected
    
    def check(self, text):
        """Main checking function"""
        if not text.strip():
            return [], text
        
        # Find errors
        errors = self.find_errors(text)
        
        # Correct text
        corrected = self.correct_text(text, errors)
        
        return errors, corrected
    
    def calculate_score(self, text, errors):
        """Calculate grammar score"""
        if not text.strip() or len(errors) == 0:
            return 100
        
        word_count = len(text.split())
        if word_count == 0:
            return 100
        
        # Start with perfect score
        score = 100
        
        # Deduct for errors
        for error in errors:
            if error['severity'] == 'high':
                score -= 5
            elif error['severity'] == 'medium':
                score -= 3
            else:
                score -= 1
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        return score

# Create checker
checker = GrammarChecker()

@app.route('/api/check', methods=['POST'])
def check_grammar():
    """API endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': True,
                'errors': [],
                'corrected_text': '',
                'score': 100,
                'word_count': 0
            })
        
        # Check grammar
        errors, corrected = checker.check(text)
        score = checker.calculate_score(text, errors)
        
        response = {
            'success': True,
            'errors': errors,
            'corrected_text': corrected,
            'score': score,
            'word_count': len(text.split()),
            'error_count': len(errors)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'running',
        'models_loaded': len(MODELS),
        'language_tool': tool is not None,
        'version': '1.0'
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Grammar Checker Started")
    print("="*50)
    print(f"📊 Models: {len(MODELS)} loaded")
    print(f"🌐 Server: http://localhost:5000")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)