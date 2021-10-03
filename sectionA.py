import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv(r"F:\Semester 7\train_u6lujuX_CVtuZ9i.csv")
test=pd.read_csv(r"F:\Semester 7\test_Y3wMUE5_7gLdaTN.csv")

test.head()
train.columns
test.columns
train.dtypes

print("Train dataset shape ", train.shape)
train.head()

print('Test dataset shape', test.shape)
test.head()

train["Loan_Status"].count()
train['Loan_Status'].value_counts(normalize=True)*100

train['Loan_Status'].value_counts(normalize=True).plot.bar(title='Loan Status')
plt.ylabel('Loan Applicants ')
plt.xlabel('Status')
plt.show()

print("The loan of 422(around 69%) people out of 614 was approved.")

print("Categorical features: These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)")

train["Gender"].count()
train['Gender'].value_counts()
(train['Gender'].value_counts(normalize=True)*100)

print("Question 1(a): Find out the number of male and female in loan applicants data.")
train['Gender'].value_counts(normalize=True).plot.bar(title ='Gender of loan applicant data')
print(train['Gender'].value_counts(normalize=True)*100)
plt.xlabel('Gender')
plt.ylabel('Number of loan applicant')
plt.show()


print(train["Married"].count())
print(train["Married"].value_counts())
train['Married'].value_counts(normalize=True)*100

print("Question 1(b) Find out the number of married and unmarried loan applicants.")
train['Married'].value_counts(normalize=True).plot.bar(title='Married Status of an applicant')
print('yes means married and no means unmarried')
print(train['Married'].value_counts(normalize=True)*100)
plt.xlabel('Married Status')
plt.ylabel('Number of loan applicant')
plt.show()

train["Self_Employed"].count()
train['Self_Employed'].value_counts()
train['Self_Employed'].value_counts(normalize=True)*100

print("Question 1 (c) Find out the overall dependent status in the dataset.")
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Dependent Status')
print(train['Self_Employed'].value_counts(normalize=True)*100)
plt.xlabel('Dependent Status')
plt.ylabel('Number of loan applicant')
plt.show()

train["Education"].count()
train["Education"].value_counts()
train["Education"].value_counts(normalize=True)*100

print(" Question 1(d) Find the count how many loan applicants are graduate and non graduate.")
train["Education"].value_counts(normalize=True).plot.bar(title = "Education")
print(train["Education"].value_counts(normalize=True)*100)
plt.xlabel('Dependent Status')
plt.ylabel('Percentage')
plt.show()

train["Property_Area"].count()
train["Property_Area"].value_counts()
train["Property_Area"].value_counts(normalize=True)*100

print("Question 1(e) Find out the count how many loan applicants property lies in urban, rural and semi-urban areas.")
train["Property_Area"].value_counts(normalize=True).plot.bar(title="Property_Area")
print(train["Property_Area"].value_counts(normalize=True)*100)
#plt.xlabel('Dependent Status')
plt.ylabel('Percentage')
plt.show()

train["Credit_History"].count()
train["Credit_History"].value_counts()
train['Credit_History'].value_counts(normalize=True)*100

train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
print(train['Credit_History'].value_counts(normalize=True)*100)
plt.xlabel('Debt')
plt.ylabel('Percentage')
plt.show()

print("Question 3")
print("To visualize and plot the distribution plot of all numerical attributes of the given train dataset i.e. ApplicantIncome,  CoApplicantIncome and LoanAmount.     ")

print("ApplicantIncome distribution: ")
print(train["ApplicantIncome"])
plt.figure(1)
plt.subplot(121)
sns.distplot(train["ApplicantIncome"]);

plt.subplot(122)
train["ApplicantIncome"].plot.box(figsize=(16,5))
plt.show()

train.boxplot(column='ApplicantIncome',by="Education" )
plt.suptitle(" ")
plt.show()

print("Coapplicant Income distribution")
plt.figure(1)
plt.subplot(121)
sns.distplot(train["CoapplicantIncome"]);

plt.subplot(122)
train["CoapplicantIncome"].plot.box(figsize=(16,5))
plt.show()

print("Loan Amount Variable")
plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(df['LoanAmount']);

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))

plt.show()
print("Loan Amount Term Distribution")
plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.distplot(df["Loan_Amount_Term"]);

plt.subplot(122)
df["Loan_Amount_Term"].plot.box(figsize=(16,5))
plt.show()



