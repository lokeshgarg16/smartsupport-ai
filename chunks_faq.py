import json

with open("airtel_dth_faqs.json") as f:
    data = json.load(f)

faqs = data["faq"]

chunks = [f"Q: {faq['question']} A: {faq['answer']}" for faq in faqs]

with open("faq_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)
