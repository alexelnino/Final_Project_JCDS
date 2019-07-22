# Load the necessary packAs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

df = pd.read_csv(
    'diabetes.csv',
    names = ['P','G','BP','ST','I','BMI','A','DPF','O'],
    header = 0
)

plt.figure(figsize=(10,6))
plt.subplot(2,4,1)
plt.bar(df['O'],df['P'], color=['green'])
plt.ylabel('Pregnancies')
plt.xlabel('Outcome')
plt.xticks([0,1])

plt.subplot(2,4,2)
plt.bar(df['O'],df['G'], color=['green'])
plt.ylabel('Glucose')
plt.xlabel('Outcome')
plt.xticks([0,1])

plt.subplot(2,4,3)
plt.bar(df['O'],df['BP'], color=['green'])
plt.ylabel('Blood Pressure')
plt.xlabel('Outcome')
plt.xticks([0,1])

plt.subplot(2,4,4)
plt.bar(df['O'],df['ST'], color=['green'])
plt.ylabel('Skin Thickness')
plt.xlabel('Outcome')
plt.xticks([0,1])

plt.subplot(2,4,5)
plt.bar(df['O'],df['I'], color=['green'])
plt.ylabel('Insulin')
plt.xlabel('Outcome')
plt.xticks([0,1])

plt.subplot(2,4,6)
plt.bar(df['O'],df['BMI'], color=['green'])
plt.ylabel('BMI')
plt.xlabel('Outcome')
plt.xticks([0,1])

plt.subplot(2,4,7)
plt.bar(df['O'],df['P'], color=['green'])
plt.ylabel('Diabetes Pedigree Function')
plt.xlabel('Outcome')
plt.xticks([0,1])

plt.subplot(2,4,8)
plt.bar(df['O'],df['P'], color=['green'])
plt.ylabel('Age')
plt.xlabel('Outcome')
plt.xticks([0,1])

plt.grid(True)
plt.tight_layout()
plt.savefig('graphical.png')
plt.show()
