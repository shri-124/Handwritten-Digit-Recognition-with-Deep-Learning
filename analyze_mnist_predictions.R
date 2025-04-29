# Load libraries
library(tidyverse)
library(caret)

# Load predictions
predictions <- read.csv("mnist_predictions.csv")

# View structure
str(predictions)

# Create confusion matrix
cm <- confusionMatrix(
  as.factor(predictions$predicted_label),
  as.factor(predictions$true_label)
)
print(cm)

# Make binary versions of the predicted and true labels
pred_binary <- factor(ifelse(predictions$predicted_label == 1, "1", "0"))
true_binary  <- factor(ifelse(predictions$true_label == 1, "1", "0"))

# Now calculate precision, recall, and F1
precision <- posPredValue(pred_binary, true_binary, positive = "1")
recall <- sensitivity(pred_binary, true_binary, positive = "1")
f1_score <- (2 * precision * recall) / (precision + recall)


cat(sprintf("Precision: %.2f%%\nRecall: %.2f%%\nF1 Score: %.2f%%\n", precision*100, recall*100, f1_score*100))

# Plot per-class accuracy
class_accuracy <- predictions %>%
  mutate(correct = true_label == predicted_label) %>%
  group_by(true_label) %>%
  summarise(accuracy = mean(correct) * 100)

ggplot(class_accuracy, aes(x = factor(true_label), y = accuracy)) +
  geom_col(fill = "skyblue") +
  labs(title = "Per-Class Accuracy", x = "Digit Label", y = "Accuracy (%)") +
  theme_minimal()
