import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

columns = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_semicolon", "char_freq_paren", "char_freq_bracket",
    "char_freq_exclaim", "char_freq_dollar", "char_freq_hash",
    "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total", "spam"
]

df = pd.read_csv("/workspaces/math3180/midterm-project/spambase/spambase.data", header=None, names=columns)

X = df.drop(columns="spam")
y = df["spam"]

X_binary = (X > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
   X_binary, y, test_size=0.3, random_state=42, stratify=y
)


df["spam"].value_counts().sort_index().plot(kind="bar")
plt.xticks([0,1],["Non-Spam","Spam"],rotation=0)
plt.title("Class Distribution")
plt.show()

features = ["word_freq_free","word_freq_money","char_freq_dollar"]

for col in features:
   prop = df.groupby("spam")[col].apply(lambda x: (x>0).mean())
   prop.plot(kind="bar")
   plt.title(f"Proportion of Emails Containing {col}")
   plt.show()



bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1:",f1_score(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png")
plt.close()
