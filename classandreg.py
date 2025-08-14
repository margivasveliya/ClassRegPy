import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import graphviz


iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)


dot_data = export_graphviz(
    full_tree,
    out_file=None,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("full_decision_tree")  

print("Full depth decision tree saved as 'full_decision_tree.pdf'")


shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
shallow_tree.fit(X_train, y_train)


y_pred_full = full_tree.predict(X_test)
y_pred_shallow = shallow_tree.predict(X_test)


acc_full = accuracy_score(y_test, y_pred_full)
acc_shallow = accuracy_score(y_test, y_pred_shallow)

print(f"Accuracy of full depth tree: {acc_full:.4f}")
print(f"Accuracy of shallow tree (max_depth=3): {acc_shallow:.4f}")


rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest accuracy: {acc_rf:.4f}")


importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(len(importances)), importances[indices], color="skyblue")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

cv_scores_tree = cross_val_score(shallow_tree, X, y, cv=5)
cv_scores_rf = cross_val_score(rf_clf, X, y, cv=5)

print("Cross-validation results (5-fold):")
print(f"Decision Tree CV accuracy: {cv_scores_tree.mean():.4f}")
print(f"Random Forest CV accuracy: {cv_scores_rf.mean():.4f}")
