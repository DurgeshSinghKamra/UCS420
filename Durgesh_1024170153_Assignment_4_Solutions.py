# UCS420: Cognitive Computing
# Assignment 4 - A Cognitive FAQ System Using Pandas (Nova 2.0)
#
# The assignment asks for a personalized 6-entry FAQ DataFrame, a simple
# confidence-scoring function, category filtering, keyword updating and CSV
# export, category counts, and explicit handling of tied highest scores.
#
# Run this file and enter your college roll number when prompted.

import re
import pandas as pd


# Q1. Build Your Personalized Knowledge Base:
# Create exactly 6 FAQ entries: 4 fixed entries and 2 entries based on the
# last two digits of the student's roll number.

roll_number = input("Enter your college roll number: ").strip()
digits = re.findall(r"\d", roll_number)

if len(digits) < 2:
    raise ValueError("The roll number must contain at least two digits.")

last_two_digits = [int(digits[-2]), int(digits[-1])]
categories = ["billing", "account", "general"]

fixed_entries = [
    {
        "question": "what is the annual fee",
        "answer": "The annual fee is Rs 500.",
        "keywords": "fee cost price charge",
        "category": "billing",
    },
    {
        "question": "how to reset password",
        "answer": "Go to Settings > Reset Password.",
        "keywords": "password reset login",
        "category": "account",
    },
    {
        "question": "what are your working hours",
        "answer": "We are open 9 AM to 5 PM.",
        "keywords": "hours timing open time",
        "category": "general",
    },
    {
        "question": "how can i pay the fee",
        "answer": "You can pay via UPI, card, or net banking.",
        "keywords": "pay payment upi fee",
        "category": "billing",
    },
]

# Each final digit determines the category using digit % 3.
personalized_entries = []
personalized_templates = {
    "billing": {
        "question": "how can i check my pending payment",
        "answer": "Open the billing section of your account to view pending payments.",
        "keywords": "pending payment dues billing",
    },
    "account": {
        "question": "how do i update my registered mobile number",
        "answer": "Open account settings, choose personal details, and update your mobile number.",
        "keywords": "mobile number update account",
    },
    "general": {
        "question": "where can i find help with a general query",
        "answer": "Contact the support desk during working hours for general assistance.",
        "keywords": "help support query assistance",
    },
}

for digit in last_two_digits:
    category = categories[digit % 3]
    entry = personalized_templates[category].copy()
    entry["question"] = entry["question"] + f" (personalized entry for digit {digit})"
    personalized_entries.append(entry)

faq_df = pd.DataFrame(fixed_entries + personalized_entries)

print("\nQ1. Final 6-row FAQ DataFrame:")
print(faq_df.to_string(index=False))


# Q2. Generate and Score a Hypothesis:
# This scoring function compares the query words with each entry's question
# and keywords. Confidence is the percentage of query words that match.
# All entries with a score greater than zero are returned, highest score first.

def score_faq(query, df):
    query_words = set(re.findall(r"\b\w+\b", query.lower()))
    # Common words add little meaning, so ignore them during matching.
    stop_words = {
        "a", "an", "the", "is", "are", "what", "how", "can", "i",
        "do", "my", "to", "of", "for", "in", "where", "with", "and"
    }
    query_words -= stop_words

    results = []
    for index, row in df.iterrows():
        searchable_text = str(row["question"]) + " " + str(row["keywords"])
        entry_words = set(re.findall(r"\b\w+\b", searchable_text.lower()))
        entry_words -= stop_words
        matched_words = query_words & entry_words

        if query_words:
            confidence = len(matched_words) / len(query_words)
        else:
            confidence = 0.0

        if confidence > 0:
            results.append({
                "index": index,
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "matched_keywords": ", ".join(sorted(matched_words)),
                "confidence": round(confidence, 2),
            })

    results.sort(key=lambda item: item["confidence"], reverse=True)
    return pd.DataFrame(results)


query = input("\nQ2. Enter a question to search the FAQ: ").strip()
matches = score_faq(query, faq_df)

if matches.empty:
    print("No matching FAQ entries were found.")
else:
    print("Matching FAQ entries ranked by confidence:")
    print(matches.to_string(index=False))


# Q3. Write same_category(category_name, df) to return all questions in a
# given category. Demonstrate it using the category of a personalized entry.

def same_category(category_name, df):
    return df[df["category"].str.lower() == category_name.lower()]

personalized_category = personalized_entries[0]["category"]
print(f"\nQ3. FAQ entries in personalized category '{personalized_category}':")
print(same_category(personalized_category, faq_df).to_string(index=False))


# Q4. Ask the user for a new keyword, add it to one FAQ entry, and save the
# complete updated DataFrame as <roll_number>_faq_data.csv.

print("\nQ4. Choose an FAQ entry to update:")
for index, row in faq_df.iterrows():
    print(f"{index}: {row['question']}")

try:
    entry_index = int(input("Enter the index of the entry to update (0-5): "))
    if entry_index not in faq_df.index:
        print("Invalid index. No keyword was added.")
    else:
        new_keyword = input("Enter a new keyword to add: ").strip()
        if new_keyword:
            existing_keywords = str(faq_df.at[entry_index, "keywords"]).split()
            if new_keyword.lower() not in [word.lower() for word in existing_keywords]:
                faq_df.at[entry_index, "keywords"] = (
                    str(faq_df.at[entry_index, "keywords"]) + " " + new_keyword
                ).strip()
            print("Updated entry:")
            print(faq_df.loc[[entry_index]].to_string(index=False))
        else:
            print("No keyword entered; the keywords were left unchanged.")
except ValueError:
    print("Invalid input. Please enter an integer index.")

# Keep the filename safe while preserving the roll number entered by the user.
safe_roll_number = re.sub(r"[^A-Za-z0-9_-]", "_", roll_number)
output_filename = f"{safe_roll_number}_faq_data.csv"
faq_df.to_csv(output_filename, index=False)
print(f"Updated FAQ DataFrame saved to {output_filename}")


# Q5. Use groupby to print the number of FAQ entries in each category.

print("\nQ5. Number of FAQ entries per category:")
print(faq_df.groupby("category").size())


# Q6. Modify the scoring function so that all entries tied for the highest
# score are printed. Demonstrate one query that produces a tie and one query
# that does not. The function below uses the same scoring function as Q2.

def show_best_matches(query_text, df):
    results = score_faq(query_text, df)

    if results.empty:
        print(f"No matching FAQ entries for: {query_text!r}")
        return

    highest_score = results["confidence"].max()
    best_matches = results[results["confidence"] == highest_score]

    print(f"\nQuery: {query_text!r}")
    print(f"Highest confidence score: {highest_score:.2f}")

    if len(best_matches) > 1:
        print("Tie found. All equally best matching entries are shown:")
    else:
        print("No tie. Best matching entry:")

    print(best_matches.to_string(index=False))


# This query matches both fixed billing FAQs through the shared word 'fee'.
show_best_matches("fee", faq_df)

# This query is intended to match the password-reset FAQ most strongly.
show_best_matches("reset password login", faq_df)
