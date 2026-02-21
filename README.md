# Homework-2q5.3

👩‍🎓 Student Information

      Name: MOPARTHI APARNA
      Course: CS5760 Natural Language Processing
      Department: Computer Science & Cybersecurity
      Semester: Spring 2026
      Assignment: Homework 2, Question 5.3

📚 Overview

**Evaluation Metrics from a Multi‑Class Confusion Matrix**

  In multi‑class classification, we need to measure how well our model performs across all classes. A confusion matrix gives a complete picture: rows represent the predicted labels, columns the true labels. From this matrix we can derive    several important metrics.
  
  1. Per‑Class Precision and Recall
     
        For each class, we treat it as the “positive” class and all other classes as “negative”. We then compute:
        
          True Positives (TP): the number of times the class was correctly predicted (diagonal element).
          False Positives (FP): the sum of the column for that class minus TP (i.e., other rows predicted this class but the true label was something else).
          False Negatives (FN): the sum of the row for that class minus TP (i.e., instances of this class that were misclassified as another class).
        
        Then:
        
          Precision = TP / (TP + FP)
          Out of all items predicted as class X, how many actually belong to class X?
          
          Recall = TP / (TP + FN)
          Out of all true items of class X, how many were correctly predicted?
        
        These two metrics often trade off against each other. A perfect classifier would have both equal to 1 for every class.
  
  2. Macro‑Averaging
     
        Macro‑averaging computes the metric independently for each class and then takes the simple average. It treats all classes equally, regardless of how many instances they have.
        
          Macro‑Precision = (P₁ + P₂ + … + Pₖ) / k
          Macro‑Recall = (R₁ + R₂ + … + Rₖ) / k
        
        Macro‑averaging is useful when you care about performance on minority classes as much as on majority classes. It can reveal if the model is good only on frequent classes.
  
  3. Micro‑Averaging
     
        Micro‑averaging aggregates the counts of TP, FP, and FN over all classes and then computes the metric from those totals.
        
          Total TP = sum of all diagonal elements
          Total FP = sum of all column sums minus total TP (or equivalently sum of all off‑diagonal column entries)
          Total FN = sum of all row sums minus total TP (sum of all off‑diagonal row entries)
        
        Then:
        
          Micro‑Precision = Total TP / (Total TP + Total FP)
          Micro‑Recall = Total TP / (Total TP + Total FN)
        
        Because of the way these totals work out, micro‑precision and micro‑recall are always equal and give the overall accuracy of the classifier. Micro‑averaging weights each instance equally, so larger classes dominate the score.
  
  4. Which One to Use?
     
        Micro‑averaging is appropriate when you care about overall correctness across all decisions (e.g., document classification where each document is equally important).
        Macro‑averaging is better when you want to ensure that performance on each class is considered fairly, even if some classes are rare (e.g., detecting rare diseases).
        
        Often both are reported to give a complete picture.
        
          Example (from Q5)
                      True Cat	True Dog	True Rabbit
          Pred Cat	  5	        10	         5
          Pred Dog	  15	    20      	10
          Pred Rabbit  0	        15	        10
     
            Class Cat: TP=5, FP=15, FN=15 → P=5/20=0.25, R=5/20=0.25
            Class Dog: TP=20, FP=25, FN=25 → P=20/45≈0.444, R=20/45≈0.444
            Class Rabbit: TP=10, FP=15, FN=15 → P=10/25=0.4, R=10/25=0.4
     
            Macro‑P = (0.25+0.444+0.4)/3 ≈ 0.365
            Micro‑P = (5+20+10)/(90) = 35/90 ≈ 0.389 (same as accuracy)
        
        This example shows that macro‑averaging gives a lower score because it penalises the poor performance on the minority class (Cat), while micro‑averaging is pulled up by the larger Dog class.

This Python script evaluates a multi‑class confusion matrix provided interactively by the user.

  It calculates:
  
    Per‑class Precision and Recall
    Macro‑averaged Precision and Recall
    Micro‑averaged Precision and Recall (including micro F1‑score)

The program prompts the user to:

    1. Enter the number of classes.
    2. Enter a name for each class.
    3. Enter the confusion matrix row by row (space‑separated integers).
    4. Rows represent predicted labels, Columns represent true labels.
    5. After receiving the matrix, the script computes and displays:
    6. Precision and recall for each class.
    7. Macro‑averaged precision and recall (simple average over classes).
    8. Micro‑averaged precision and recall (aggregated over all instances).
    9. Micro F1‑score (if defined).

🚀 Requirements

    Python 3.6 or higher
    NumPy – Install via pip install numpy

📝 How to Run

    Clone the repository or download q5.3.py.
    Open a terminal in the directory containing the script.
    
    Run the command:
    
    bash
    python q5.3.py
    Follow the interactive prompts and enter the confusion matrix when asked.
    
📊 Example Output
      === Confusion Matrix Evaluation ===

      Enter number of classes: 3
      Enter name for class 1: Cat
      Enter name for class 2: Dog
      Enter name for class 3: Rabbit
      
      Enter the confusion matrix row by row (space-separated values).
      Rows = predicted, Columns = true.
      Row 1 (Cat): 5 10 5
      Row 2 (Dog): 15 20 10
      Row 3 (Rabbit): 0 15 10
      
      --- Per-Class Metrics ---
      Cat   : Precision = 0.2500, Recall = 0.2500
      Dog   : Precision = 0.4444, Recall = 0.4444
      Rabbit: Precision = 0.4000, Recall = 0.4000
      
      Macro-averaged Precision = 0.3648
      Macro-averaged Recall    = 0.3648
      
      Micro-averaged Precision = 0.3889
      Micro-averaged Recall    = 0.3889
      (Micro-averaged F1 = 0.3889)

    
🔧 Code Structure

    get_confusion_matrix_from_user() – Handles interactive input, validates the matrix dimensions, and returns the matrix and class names.
    
    compute_metrics(conf_matrix, class_names) – Accepts a 2D list/array and class names, computes all metrics, and prints them clearly.
    
    The script runs in __main__ by calling these two functions.

📈 Metrics Definitions

    Precision = TP / (TP + FP) – Of the items predicted as class *c*, how many actually belong to *c*?
    
    Recall = TP / (TP + FN) – Of the items that truly belong to class *c*, how many were correctly predicted?
    
    Macro‑average – Simple average of per‑class metrics (treats all classes equally).
    
    Micro‑average – Aggregates TP, FP, FN over all classes before computing the metric (weights larger classes more).
    
    Micro F1 = 2 * (P<sub>micro</sub> * R<sub>micro</sub>) / (P<sub>micro</sub> + R<sub>micro</sub>)
