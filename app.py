from datasets import load_dataset
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
import evaluate

# ✅ Step 1: Load dataset
dataset = load_dataset("SetFit/amazon_reviews_multi_en")


# ✅ Step 2: Tokenizer
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize, batched=True)

# ✅ Step 3: Load model
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=5)

# ✅ Step 4: Metric
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# ✅ Step 5: Training Arguments — load_best_model_at_end removed
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",  # ✅ saves model at end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
)

# ✅ Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=tokenized["validation"].select(range(500)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ✅ Step 7: Train
trainer.train()

# ✅ Step 8: Evaluate manually on test set
results = trainer.evaluate(tokenized["test"].select(range(500)))
print("Test Accuracy:", results["eval_accuracy"])

# ✅ Step 9: Save model
model.save_pretrained("./my_sentiment_model")
tokenizer.save_pretrained("./my_sentiment_model")
