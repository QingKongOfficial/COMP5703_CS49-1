STEP1: use the environment.txt to install the pack needed.

pip install environment.txt

STEP2: use the embedding.py to build the sector database

python embedding.py

STEP3: use the ReRAG.py to processe the training data

python ReRAG.py \
    --input_file "flashcards_train.json" \
    --output_file "train_processed.json"

STEP4: use the rsDora+ method and the training data to build the generator model

STEP5: use the ReRAG.py to process the testing data

python ReRAG.py \
    --input_file "flashcards_test.json" \
    --output_file "test_processed.json"

STEP6: generate output using testing data and evaluate the output

@article{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Yongqiang Ma},
  journal={arXiv preprint arXiv:2403.13372},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
