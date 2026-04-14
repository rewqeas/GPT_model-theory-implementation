from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_ds = dataset['train']['text']
val_ds = dataset['validation']['text']
test_ds = dataset['test']['text']

train_text = "\n".join(train_ds)
val_text = "\n".join(val_ds)
test_text = "\n".join(test_ds)


with open("wikitext2_train.txt", 'w', encoding="utf-8") as f:
    f.write(train_text)

with open("wikitext2_val.txt", 'w', encoding="utf-8") as f:
    f.write(val_text)

with open("wikitext2_test.txt", 'w', encoding="utf-8") as f:
    f.write(test_text)