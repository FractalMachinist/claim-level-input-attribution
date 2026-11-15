# Setup and data-fetch steps. The datasets themselves are gitignored (data/);
# this file is the reproducible way to recreate them.

SIMPLEQA_URL = https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv
SQUAD_DEV_URL = https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
SQUAD_TRAIN_URL = https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

.PHONY: data simpleqa squad clean-data

## data: fetch all benchmark datasets into data/ (gitignored)
data: simpleqa squad

## simpleqa: OpenAI SimpleQA test set (4,326 closed-book factual QAs) -- Stage 1
simpleqa: data/simpleqa/simple_qa_test_set.csv
data/simpleqa/simple_qa_test_set.csv:
	mkdir -p data/simpleqa
	curl -sL -o $@ "$(SIMPLEQA_URL)"
	@echo "fetched SimpleQA: $$(wc -l < $@) lines"

## squad: SQuAD v1.1 dev set (10,570 extractive QAs with answer spans) -- Stage 2
squad: data/squad/dev-v1.1.json
data/squad/dev-v1.1.json:
	mkdir -p data/squad
	curl -sL -o $@ "$(SQUAD_DEV_URL)"
	@echo "fetched SQuAD dev: $$(du -h $@ | cut -f1)"

## squad-train: SQuAD v1.1 train set (larger; optional backup)
squad-train: data/squad/train-v1.1.json
data/squad/train-v1.1.json:
	mkdir -p data/squad
	curl -sL -o $@ "$(SQUAD_TRAIN_URL)"

## clean-data: remove fetched datasets
clean-data:
	rm -rf data/
