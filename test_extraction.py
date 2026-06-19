import sys, os
sys.path.append(r'c:/Users/knowu/Documents/Projects/AI_Hackathon/vector-cartel/src')
import ollama_inference

md_path = r'c:/Users/knowu/Documents/Projects/AI_Hackathon/vector-cartel/notebook/corpus/advisories/ICSA-26-167-02.md'
pdf_path = r'c:/Users/knowu/Documents/Projects/AI_Hackathon/vector-cartel/notebook/corpus/nist_sp800_82r3.pdf'
md1_path = r'c:/Users/knowu/Documents/Projects/AI_Hackathon/vector-cartel/notebook/corpus/attack/T0811.md'

print('MD metadata:', ollama_inference.extract_metadata_from_corpus(md_path))
print('--------------------------------------------------\n')
# print('PDF metadata:', ollama_inference.extract_metadata_from_corpus(pdf_path))
# print('--------------------------------------------------\n')
print('MD metadata:', ollama_inference.extract_metadata_from_corpus(md1_path))
