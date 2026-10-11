import pandas as pd

# Q1. Create a DataFrame using the table given in the assignment.
# The columns are: TID, Refund, Marital Status, Taxable Income, Cheat.

data = {
    "TID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Refund": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No"],
    "Marital Status": [
        "Single", "Married", "Single", "Married", "Divorced",
        "Married", "Divorced", "Single", "Married", "Single"
    ],
    "Taxable Income": [
        "125K", "100K", "70K", "120K", "95K",
        "60K", "220K", "85K", "75K", "90K"
    ],
    "Cheat": ["No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)
print("Q1. Dataset:")
print(df)


# Q2. From the table created above, locate rows 0, 4, 7 and 8 using DataFrame.

print("\nQ2. Rows with index 0, 4, 7 and 8:")
print(df.loc[[0, 4, 7, 8]])


# Q3. Navigate the DataFrame and perform the following tasks.

# Q3(a). Select rows from index 3 to 7.
# loc includes both endpoints, so rows 3, 4, 5, 6 and 7 are selected.
print("\nQ3(a). Rows from index 3 to 7:")
print(df.loc[3:7])

# Q3(b). Select rows from index 4 to 8, and columns 2 to 4.
# The column positions are zero-based. iloc includes the start but excludes
# the stop, so use iloc[4:9, 2:5] to include row index 8 and column position 4.
print("\nQ3(b). Rows 4 to 8 and columns 2 to 4:")
print(df.iloc[4:9, 2:5])

# Q3(c). Select all rows with column index 1 to 3, including column index 3.
# iloc uses an exclusive stop, so columns at positions 1, 2 and 3 are selected.
print("\nQ3(c). All rows and columns at positions 1 to 3:")
print(df.iloc[:, 1:4])


# Q4. Read a CSV file and display its first five rows.
# Download the Iris dataset from the URL in the assignment and save it as
# 'iris.csv' in the same folder as this Python file before running this section.
# If the file is named differently, update the filename below.

try:
    iris_df = pd.read_csv("iris.csv")
    print("\nQ4. First five rows of iris.csv:")
    print(iris_df.head())
except FileNotFoundError:
    print("\nQ4. Could not find 'iris.csv'. Download the Iris CSV and place it "
          "in the same folder as this Python file to run Q4 and Q5.")


# Q5. From the CSV file used in Q4, delete row 4 and delete column 3.
# Row index 4 and column position 3 are zero-based.
# This section runs if iris.csv was loaded successfully.

if "iris_df" in globals():
    iris_modified = iris_df.drop(index=4)
    column_to_delete = iris_modified.columns[3]
    iris_modified = iris_modified.drop(columns=[column_to_delete])
    print("\nQ5. Iris dataset after deleting row index 4 and column position 3:")
    print(iris_modified)


# Q6. Create employees.csv using the employee data given in the assignment,
# then use Pandas to perform the required operations (a) to (l).

employee_data = {
    "Employee_ID": [101, 102, 103, 104, 105],
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Edward"],
    "Department": ["HR", "IT", "IT", "Marketing", "Sales"],
    "Age": [29, 34, 41, 28, 38],
    "Salary": [50000, 70000, 65000, 55000, 60000],
    "Years_of_Experience": [4, 8, 10, 3, 12],
    "Joining_Date": ["2020-03-15", "2017-07-19", "2013-06-01",
                     "2021-02-10", "2010-11-25"],
    "Gender": ["Female", "Male", "Male", "Female", "Male"],
    "Bonus": [5000, 7000, 6000, 4500, 5000],
    "Rating": [4.5, 4.0, 3.8, 4.7, 3.5]
}

employees = pd.DataFrame(employee_data)

# Save the original sample dataset as employees.csv.
employees.to_csv("employees.csv", index=False)
print("\nQ6. Employee dataset saved as employees.csv:")
print(employees)


# Q6(a). Display the shape (number of rows and columns) of the DataFrame.
print("\nQ6(a). Shape (rows, columns):")
print(employees.shape)


# Q6(b). Display data types and non-null counts for every column.
print("\nQ6(b). DataFrame summary:")
employees.info()


# Q6(c). Generate descriptive statistics.
print("\nQ6(c). Descriptive statistics:")
print(employees.describe())


# Q6(d). Display the first five rows and the last three rows.
print("\nQ6(d). First five rows:")
print(employees.head(5))
print("Last three rows:")
print(employees.tail(3))


# Q6(e). Calculate average salary, total bonus, youngest age, and highest rating.
print("\nQ6(e)(i). Average salary:", employees["Salary"].mean())
print("Q6(e)(ii). Total bonus paid:", employees["Bonus"].sum())
print("Q6(e)(iii). Youngest employee's age:", employees["Age"].min())
print("Q6(e)(iv). Highest performance rating:", employees["Rating"].max())


# Q6(f). Sort the DataFrame by Salary in descending order.
employees = employees.sort_values(by="Salary", ascending=False)
print("\nQ6(f). Employees sorted by salary (highest first):")
print(employees)


# Q6(g). Add a Performance_Category column based on the employee rating.
def performance_category(rating):
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 4.0:
        return "Good"
    else:
        return "Average"


employees["Performance_Category"] = employees["Rating"].apply(performance_category)
print("\nQ6(g). DataFrame with performance categories:")
print(employees[["Name", "Rating", "Performance_Category"]])


# Q6(h). Identify missing values in the DataFrame.
print("\nQ6(h). Missing values in each column:")
print(employees.isnull().sum())


# Q6(i). Rename Employee_ID to ID.
employees = employees.rename(columns={"Employee_ID": "ID"})
print("\nQ6(i). DataFrame after renaming Employee_ID to ID:")
print(employees.head())


# Q6(j)(i). Find employees with more than five years of experience.
print("\nQ6(j)(i). Employees with more than 5 years of experience:")
print(employees[employees["Years_of_Experience"] > 5])


# Q6(j)(ii). Find employees who belong to the IT department.
print("\nQ6(j)(ii). Employees in the IT department:")
print(employees[employees["Department"] == "IT"])


# Q6(k). Add a Tax column that deducts 10% of Salary.
# Here Tax represents the amount deducted (10% of salary), not salary after deduction.
employees["Tax"] = employees["Salary"] * 0.10
print("\nQ6(k). DataFrame with Tax column:")
print(employees[["Name", "Salary", "Tax"]])


# Q6(l). Save the modified DataFrame, including the added columns, to a new CSV file.
employees.to_csv("employees_modified.csv", index=False)
print("\nQ6(l). Modified dataset saved as employees_modified.csv")
