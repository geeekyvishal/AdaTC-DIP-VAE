import json

with open("proposal_output.json") as f:
    data = json.load(f)

result = data["result"]
required_keys = [
    "candidate_name",
    "objective_latex",
    "derivation",
    "gaps_addressed",
    "why_reconstruction_preserved",
    "gradient_tractability",
    "hyperparameters",
    "limitations",
    "citations"
]

for k in required_keys:
    assert k in result, f"Missing key {k}"
    print(f"Key '{k}': {type(result[k])}, length/size = {len(result[k])}")

# Check that every gap is addressed
assert len(result["gaps_addressed"]) == 5, f"Expected 5 gaps, got {len(result['gaps_addressed'])}"

# Check that citations are valid paper IDs from catalog
with open("research-paper/catalog.csv") as f:
    lines = f.readlines()[1:]
valid_ids = [l.split(",")[0].strip() for l in lines]
print("Valid IDs from catalog:", valid_ids)

for cit in result["citations"]:
    paper_id = cit.split(":")[0].strip()
    assert paper_id in valid_ids, f"Invalid paper ID in citation: {paper_id}"

print("All verifications passed!")
