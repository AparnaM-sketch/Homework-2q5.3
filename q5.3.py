import numpy as np

def get_confusion_matrix_from_user():
    """Prompt user to enter the confusion matrix interactively."""
    n = int(input("Enter number of classes: "))
    class_names = []
    for i in range(n):
        name = input(f"Enter name for class {i+1}: ").strip()
        class_names.append(name)
    
    print("\nEnter the confusion matrix row by row (space-separated values).")
    print("Rows = predicted, Columns = true.")
    conf_matrix = []
    for i in range(n):
        row = list(map(int, input(f"Row {i+1} ({class_names[i]}): ").split()))
        if len(row) != n:
            print(f"Error: Row must have exactly {n} numbers. Try again.")
            return get_confusion_matrix_from_user()  # restart
        conf_matrix.append(row)
    
    return conf_matrix, class_names

def compute_metrics(conf_matrix, class_names):
    """
    Accepts a confusion matrix (2D list or numpy array) and class names,
    then prints per-class precision/recall and macro/micro averages.
    """
    conf = np.array(conf_matrix)
    n_classes = len(class_names)
    
    # Per-class TP, FP, FN
    TP = np.diag(conf)
    FP = conf.sum(axis=0) - TP   # column sum minus TP
    FN = conf.sum(axis=1) - TP   # row sum minus TP
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    # Replace NaN with 0 for classes with zero predictions/true
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    
    print("\n--- Per-Class Metrics ---")
    for i, cls in enumerate(class_names):
        print(f"{cls:6s}: Precision = {precision[i]:.4f}, Recall = {recall[i]:.4f}")
    
    # Macro averages
    macro_p = np.mean(precision)
    macro_r = np.mean(recall)
    print(f"\nMacro-averaged Precision = {macro_p:.4f}")
    print(f"Macro-averaged Recall    = {macro_r:.4f}")
    
    # Micro averages
    total_TP = np.sum(TP)
    total_FP = np.sum(FP)
    total_FN = np.sum(FN)
    micro_p = total_TP / (total_TP + total_FP)
    micro_r = total_TP / (total_TP + total_FN)
    print(f"\nMicro-averaged Precision = {micro_p:.4f}")
    print(f"Micro-averaged Recall    = {micro_r:.4f}")
    if micro_p + micro_r > 0:
        print(f"(Micro-averaged F1 = {2*micro_p*micro_r/(micro_p+micro_r):.4f})")
    else:
        print("(Micro-averaged F1 = undefined)")

if __name__ == "__main__":
    print("=== Confusion Matrix Evaluation ===\n")
    conf_matrix, class_names = get_confusion_matrix_from_user()
    compute_metrics(conf_matrix, class_names)