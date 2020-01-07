## Steps to prepare a parallel dataset for code2ast:

1) Make up a list of repositories to clone
2) Clone selected repositories
3) Parse every .py file (returning a pair of .src and .ast files) for every cloned repository
4) Merge parsed pairs into two large files (train.src, train.ast)
5) Train a BPE tokenizer model on both files (model_src, model_ast)
6) Apply tokenization for all lines in the files and filter out ones which are longer than the threshold value (512 tokens).
(This will result in making two tokenized files with lines of a length not greater than the threshold value)
7) Detokenize files using trained BPE models and write results to updated files
8) Train new BPE tokenization models on updated files
9) Tokenize updated files using new BPE models
10) Split tokenized files into train/valid/test subsets
11) Preprocess prepared subsets using fairseq-preprocess utils