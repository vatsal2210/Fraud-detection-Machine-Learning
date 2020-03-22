# Importing libraries
from tkinter import *
from collections import OrderedDict
from keras.models import model_from_json
from keras.models import load_model
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Title of GUI
master = Tk()
master.title("Credit card defaulter check")
master.geometry("300x600")
master.title('Data Analytics Project')

# Input Label
Label(master, text="Limit Balance:").grid(row=0)
Label(master, text="Male:").grid(row=1)
Label(master, text="Graduate:").grid(row=2)
Label(master, text="University:").grid(row=3)
Label(master, text="High School:").grid(row=4)
Label(master, text="Married:").grid(row=5)
Label(master, text="Single:").grid(row=6)
Label(master, text="Age:").grid(row=7)

# Payment Label
Label(master, text="Pay 1:").grid(row=8)
Label(master, text="Pay 2:").grid(row=9)
Label(master, text="Pay 3:").grid(row=10)
Label(master, text="Pay 4:").grid(row=11)
Label(master, text="Pay 5:").grid(row=12)
Label(master, text="Pay 6:").grid(row=13)

# Bill amount Label
Label(master, text="Bill Amount 1:").grid(row=14)
Label(master, text="Bill Amount 2:").grid(row=15)
Label(master, text="Bill Amount 3:").grid(row=16)
Label(master, text="Bill Amount 4:").grid(row=17)
Label(master, text="Bill Amount 5:").grid(row=18)
Label(master, text="Bill Amount 6:").grid(row=19)

# Paid Amount Label
Label(master, text="Paid Amount 1:").grid(row=20)
Label(master, text="Paid Amount 2:").grid(row=21)
Label(master, text="Paid Amount 3:").grid(row=22)
Label(master, text="Paid Amount 4:").grid(row=23)
Label(master, text="Paid Amount 5:").grid(row=24)
Label(master, text="Paid Amount 6:").grid(row=25)

# Next Month Defaulter Label
Label(master, text="Next Month Defaulter:").grid(row=29)


# Get Input values
e1 = Entry(master)
e1.grid(row=0, column=1)

e2 = Entry(master)
e2.grid(row=1, column=1)

e3 = Entry(master)
e3.grid(row=2, column=1)

e4 = Entry(master)
e4.grid(row=3, column=1)

e5 = Entry(master)
e5.grid(row=4, column=1)

e6 = Entry(master)
e6.grid(row=5, column=1)

e7 = Entry(master)
e7.grid(row=6, column=1)

e8 = Entry(master)
e8.grid(row=7, column=1)

e9 = Entry(master)
e9.grid(row=8, column=1)

e10 = Entry(master)
e10.grid(row=9, column=1)

e11 = Entry(master)
e11.grid(row=10, column=1)

e12 = Entry(master)
e12.grid(row=11, column=1)

e13 = Entry(master)
e13.grid(row=12, column=1)

e14 = Entry(master)
e14.grid(row=13, column=1)

e15 = Entry(master)
e15.grid(row=14, column=1)

e16 = Entry(master)
e16.grid(row=15, column=1)

e17 = Entry(master)
e17.grid(row=16, column=1)

e18 = Entry(master)
e18.grid(row=17, column=1)

e19 = Entry(master)
e19.grid(row=18, column=1)

e20 = Entry(master)
e20.grid(row=19, column=1)

e21 = Entry(master)
e21.grid(row=20, column=1)

e22 = Entry(master)
e22.grid(row=21, column=1)

e23 = Entry(master)
e23.grid(row=22, column=1)

e24 = Entry(master)
e24.grid(row=23, column=1)

e25 = Entry(master)
e25.grid(row=24, column=1)

e26 = Entry(master)
e26.grid(row=25, column=1)

K = Entry(master, state=DISABLED)
K.grid(row=29, column=1)

# Load Dataset
print("--- Loading Dataset ---")
url = 'H:/Western/Fall 2018/Data Analytics/Project/Code/dataset.csv'
dataset = pd.read_csv(url)

# Feature Engineering
dataset.rename(columns=lambda X: X.lower(), inplace=True)
dataset.drop('id', axis=1, inplace=True)
dataset.rename(
    columns={'default.payment.next.month': 'isDefault'}, inplace=True)

dataset['grad_school'] = (dataset['education'] == 1).astype('int')
dataset['university'] = (dataset['education'] == 2).astype('int')
dataset['high_school'] = (dataset['education'] == 3).astype('int')
dataset.drop('education', axis=1, inplace=True)

dataset['male'] = (dataset['sex'] == 1).astype('int')
dataset.drop('sex', axis=1, inplace=True)

dataset['married'] = (dataset['marriage'] == 1).astype('int')
dataset['single'] = (dataset['marriage'] == 2).astype('int')
dataset.drop('marriage', axis=1, inplace=True)

# For pay features if the <=0 then it means it was not delayed
pay_features = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
for p in pay_features:
    dataset.loc[dataset[p] <= 0, p] = 0

target_name = 'isDefault'

X = dataset.drop(target_name, axis=1)
robust_scaler = RobustScaler()
X = robust_scaler.fit_transform(X)
y = dataset[target_name]

# Train-test data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=15, stratify=y)

model = LogisticRegression(n_jobs=-1, random_state=15)

# 2. Use the training data to train the estimator
model.fit(X_train, y_train)

# 3. Evaluate the model
y_pred_test = model.predict(X_test)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_pred_proba >= 0.2).astype('int')

# Create a model and save file into a disk
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Load a file from a disk
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
	
# Call function to check result
def make_ind_prediction(data):

    a1 = float(e1.get())
    a2 = (e2.get())
    a3 = (e3.get())
    a4 = (e4.get())
    a5 = float(e5.get())
    a6 = float(e6.get())
    a7 = float(e7.get())
    a8 = float(e8.get())
    a9 = float(e9.get())
    a10 = float(e10.get())
    a11 = float(e11.get())
    a12 = float(e12.get())
    a13 = float(e13.get())
    a14 = float(e14.get())
    a15 = float(e15.get())
    a16 = float(e16.get())
    a17 = float(e17.get())
    a18 = float(e18.get())
    a19 = float(e19.get())
    a20 = float(e20.get())
    a21 = float(e21.get())
    a22 = float(e22.get())
    a23 = float(e23.get())
    a24 = float(e24.get())
    a25 = float(e25.get())
    a26 = float(e26.get())

    result = 0

    new_customer = OrderedDict([('limit_bal', a1), ('male', a2), ('grad_school', a3), ('university', a4), ('high_school', a5), ('married', a6), ('single', a7), ('age', a8), ('pay_0', a9), ('pay_2', a10), ('pay_3', a11), ('pay_4', a12), ('pay_5', a13), ('pay_6', a14),
                                ('bill_amt1', a15), ('bill_amt2', a16), ('bill_amt3', a17), ('bill_amt4', a18), ('bill_amt5', a19), ('bill_amt6', a20), ('pay_amt1', a21), ('pay_amt2', a22), ('pay_amt3', a23), ('pay_amt4', a24), ('pay_amt5', a25), ('pay_amt6', a26)])
    print(new_customer)

    new_customer = pd.Series(new_customer)
    print('in function')
    data = new_customer.values.reshape(1, -1)
    print(data)
    data = robust_scaler.transform(data)
    print(data)
    prob = model.predict_proba(data)[0][1]
    print(prob)

    K.configure(state=NORMAL)  # make the field editable

    if prob >= 0.2:
        result = 'Will default'
        print(result)
        K.insert(0, result)
        K.configure(state=DISABLED)  # make the field read only
        return
    else:
        result = 'Will pay'
        print(result)
        K.insert(0, result)
        K.configure(state=DISABLED)  # make the field read only
        return

Button(master, text='Quit', command=master.quit).grid(
    row=30, column=0, sticky=E, pady=4)
Button(master, text='Show', command=lambda: make_ind_prediction(K)).grid(
    row=30, column=1, sticky=W, pady=4)

mainloop()
