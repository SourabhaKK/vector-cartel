import sys
import os

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Dynamically add the src directory to sys.path
src_path = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(src_path)


# pyrefly: ignore [missing-import]
import ollama_inference

# Define dynamic paths for the corpus files
corpus_dir = os.path.join(PROJECT_ROOT, 'data', 'corpus')
advisory_md_path = os.path.join(corpus_dir, 'advisories', 'ICSA-26-167-02.md')
attack_md_path = os.path.join(corpus_dir, 'attack', 'T0811.md')

print('MD metadata:', ollama_inference.extract_metadata_from_corpus(advisory_md_path))
print()
print('MD metadata:', ollama_inference.extract_metadata_from_corpus(attack_md_path))
