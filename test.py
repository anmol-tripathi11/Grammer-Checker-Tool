from app import checker

# Test cases
test_cases = [
    "I will do it yesterday",
    "I can do it yesterday",
    "I should go yesterday",
    "I do it yesterday"
]

for text in test_cases:
    errors, corrected = checker.check(text)
    print(f"Input: '{text}'")
    print(f"Corrected: '{corrected}'")
    print(f"Errors: {errors}")
    print("-" * 50)
