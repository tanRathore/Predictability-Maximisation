# decoder.py
# Maps single-letter tags to human-readable linguistic terms

def decode_sequence(sequence_string):
    # Mapping based on Universal Dependencies (SUD) shorthand
    # Adjust these if your specific dataset uses different shorthand
    mapping = {
        'N': 'Noun',
        'V': 'Verb',
        'A': 'Adj',   # Adjective/Adverb
        'P': 'ADP',   # Adposition (Preposition in EN, Postposition in HI/JA)
        'D': 'Det',   # Determiner
        'R': 'Adv',   # Adverb
        'C': 'Conj',  # Conjunction
        'X': 'Other',
        '0': 'Root'   # Root placeholder
    }
    
    decoded = []
    for char in sequence_string:
        decoded.append(mapping.get(char, char))
        
    return "-".join(decoded)

# ---------------------------------------------------------
# ANALYSIS OF YOUR SPECIFIC OVERLAPS (From your logs)
# ---------------------------------------------------------
print("=== DECODING COMMON HINDI-JAPANESE STRUCTURES ===")
# These were the top overlaps you found in your logs
overlaps = ['PAPANAVAAP', 'PANNANAVAP', 'NPNNP'] 

for seq in overlaps:
    print(f"\nOriginal: {seq}")
    print(f"Decoded:  {decode_sequence(seq)}")
    
    # Heuristic Analysis
    if "V" == seq[-1]:
        print("Analysis: Strictly Verb-Final (Strong SOV trait)")
    elif "P" == seq[-1]:
        print("Analysis: Ends with Adposition (Likely Postpositional Phrase)")
    elif "V" in seq and seq.index("V") < len(seq)/2:
        print("Analysis: Verb-Initial/Medial (SVO trait)")

print("\n=== COMPARING ENGLISH VS HINDI EXAMPLES ===")
# Common English pattern (SVO) vs Hindi pattern (SOV)
examples = {
    "English (Common)": "DNVNP",  # e.g., "The dog ate near park"
    "Hindi (Common)":   "DNPV"    # e.g., "The park-in ate" (Subject dropped)
}

for lang, seq in examples.items():
    print(f"{lang}: {seq} -> {decode_sequence(seq)}")