# gtr-t5-embedding

## Code
```python
# Import necessary libraries
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sample input texts
texts = [
    "Cats are cute animals.",
    "Dogs are loyal pets.",
    "Mars is the fourth planet.",
    "Felines have sharp claws.",
    "Venus is closer to the Sun than Earth."
]

# Using 'gtr-t5-large' model as an alternative to OpenAI's 'text-embedding-ada-002'
model = SentenceTransformer('sentence-transformers/gtr-t5-large')

# Generate embeddings for each sentence
embeddings = model.encode(texts)

# Compute cosine similarity between embeddings
similarity_matrix = cosine_similarity(embeddings)

# Print the most similar sentence for each input
print("\nMost Similar Sentences:")
for i in range(len(texts)):
    similarities = similarity_matrix[i]
    most_similar_idx = np.argsort(similarities)[-2]
    print(f"'{texts[i]}' is most similar to → '{texts[most_similar_idx]}' with score {similarities[most_similar_idx]:.3f}")
```

## Output
```bash
Most Similar Sentences:
'Cats are cute animals.' is most similar to → 'Felines have sharp claws.' with score 0.709
'Dogs are loyal pets.' is most similar to → 'Cats are cute animals.' with score 0.698
'Mars is the fourth planet.' is most similar to → 'Venus is closer to the Sun than Earth.' with score 0.578
'Felines have sharp claws.' is most similar to → 'Cats are cute animals.' with score 0.709
'Venus is closer to the Sun than Earth.' is most similar to → 'Mars is the fourth planet.' with score 0.578
```
