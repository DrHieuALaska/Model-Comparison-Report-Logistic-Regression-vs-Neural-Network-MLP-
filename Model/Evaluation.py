
def evaluation(y_true, y_pred):
    from sklearn.metrics import accuracy_score

    print("Validation Accuracy:", accuracy_score(y_true, y_pred))

    # from sklearn.metrics import classification_report

    # print(classification_report(y_val, y_val_pred))

    from sklearn.metrics import confusion_matrix

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print("Recall (Sensitivity) for Malignant class:", recall)

    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    print("Precision for Malignant class:", precision)

    F1_score = 2 * (precision * recall) / (precision + recall)
    print("F1 Score for Malignant class:", F1_score)