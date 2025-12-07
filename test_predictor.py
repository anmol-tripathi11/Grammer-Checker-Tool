# Test script
from grammar_predictor import GrammarErrorPredictor

predictor = GrammarErrorPredictor()

test_cases = [
    ("He go to school every day.", "subject_verb_agreement"),
    ("She don't like apples.", "subject_verb_agreement"),
    ("Between you and I, it's wrong.", "pronoun_case"),
    ("I have saw that movie.", "irregular_verb"),
    ("A apple a day.", "article_usage"),
    ("They is coming tomorrow.", "subject_verb_agreement"),
    ("The dog chase cats.", "subject_verb_agreement"),
    ("Each student have a book.", "subject_verb_agreement"),
    ("There is many people.", "subject_verb_agreement"),
    ("Me and him went shopping.", "pronoun_case")
]

print("Testing Grammar Error Predictor:")
print("=" * 60)

correct = 0
total = len(test_cases)

for sentence, expected in test_cases:
    result = predictor.predict(sentence)
    
    if 'error' in result:
        print(f"ERROR: {result['error']}")
        continue
    
    predicted = result['error_type']
    confidence = result['confidence']
    
    matches = False
    if predicted == expected:
        matches = True
        symbol = "[CORRECT]"
    else:
        for pred in result['top_predictions']:
            if pred['error_type'] == expected:
                matches = True
                symbol = "[CLOSE]"
                predicted = f"{predicted} (expected in top predictions)"
                break
    
    if matches:
        print(f"{symbol} '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'")
        print(f"  Predicted: {predicted} ({confidence:.2%})")
        correct += 1
    else:
        print(f"[WRONG] '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'")
        print(f"  Predicted: {predicted} ({confidence:.2%})")
        print(f"  Expected: {expected}")
    
    print("-" * 60)

print(f"\nRESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
