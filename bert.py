from bert_score import score

refs = ["The cat sat on the mat."]
hyps = ["The cat sat on the mat."]

P, R, F1 = score(hyps, refs, lang="en", verbose=False)
print(f"BERTScore-F1: {F1.item():.4f}")

P, R, F1 = score(['how are you'], ['How do you do'], lang="en", verbose=False)
print(f"BERTScore-F1: {F1.item():.4f}")