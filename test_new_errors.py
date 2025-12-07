import requests
import json

# Test the new error types: passive_voice, modal_verb, subjunctive_mood

test_cases = [
    {
        "text": "The cake was eaten by John.",
        "expected_error_type": "Passive Voice"
    },
    {
        "text": "I can sings very well.",
        "expected_error_type": "Modal Verb"
    },
    {
        "text": "If I was rich, I would buy a car.",
        "expected_error_type": "Subjunctive Mood"
    },
    {
        "text": "I suggest that he goes to the store.",
        "expected_error_type": "Subjunctive Mood"
    }
]

def test_errors():
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['text']}")
        print(f"Expected Error Type: {test_case['expected_error_type']}")

        try:
            response = requests.post('http://localhost:5000/api/check',
                                   json={'text': test_case['text']},
                                   timeout=10)

            if response.status_code == 200:
                data = response.json()
                errors = data.get('errors', [])

                # Check if expected error type is found
                found_error_types = [error.get('type') for error in errors]
                print(f"Detected Error Types: {found_error_types}")

                if test_case['expected_error_type'] in found_error_types:
                    print("✅ PASS: Expected error type detected")
                else:
                    print("❌ FAIL: Expected error type not detected")

                # Print all errors for debugging
                for error in errors:
                    print(f"  - {error.get('type')}: '{error.get('text')}' -> '{error.get('correction')}'")

            else:
                print(f"❌ FAIL: API returned status {response.status_code}")

        except Exception as e:
            print(f"❌ FAIL: Exception occurred: {e}")

if __name__ == "__main__":
    test_errors()
