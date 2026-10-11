import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, RegexpStemmer, WordNetLemmatizer
from nltk import pos_tag

for resource, package in [
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True)

# Q1. Preprocessing:
# Run process_ticket() on every ticket and create a table with sentence count,
# token count, punctuation token count, stopword count, unique token count,
# and final token count. Add a total row for count-based columns. For one
# ticket, display actual tokens, punctuation tokens, stopwords, and final tokens.

def get_raw_tickets():
    path = Path("tickets.json")
    if not path.exists():
        raise FileNotFoundError(
            "Save the generated tickets from the starter notebook as tickets.json, "
            "or adapt get_raw_tickets() to use the notebook's generated variable."
        )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict) and "tickets" in loaded:
        loaded = loaded["tickets"]
    raw = []
    for item in loaded:
        if isinstance(item, str):
            raw.append(item)
        elif isinstance(item, dict):
            raw.append(str(item.get("text", item.get("ticket", item))))
        else:
            raw.append(str(item))
    return raw


tickets = get_raw_tickets()
stop_words = set(stopwords.words("english"))
word_punct_tokenizer = nltk.tokenize.TreebankWordTokenizer()


def fallback_process_ticket(text):
    # A transparent local preprocessing implementation. If the starter
    # notebook provides process_ticket(), use that function instead.
    tokens = word_punct_tokenizer.tokenize(text)
    punctuation_tokens = [token for token in tokens if re.fullmatch(r"[^\w\s]+", token)]
    stopword_tokens = [token for token in tokens if token.lower() in stop_words]
    final_tokens = [
        token.lower() for token in tokens
        if token.lower() not in stop_words and not re.fullmatch(r"[^\w\s]+", token)
    ]
    return {
        "tokens": tokens,
        "punctuation_tokens": punctuation_tokens,
        "stopwords": stopword_tokens,
        "final_tokens": final_tokens,
    }


def process_one_ticket(text):
    # Do not modify the starter's process_ticket function. If it is present in
    # the notebook, call it. Otherwise use the fallback implementation.
    starter_function = globals().get("process_ticket")
    if callable(starter_function) and starter_function is not process_one_ticket:
        return starter_function(text)
    return fallback_process_ticket(text)


def sentence_count(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return sum(bool(part) for part in parts)


def get_field(processed, possible_names, default=None):
    if isinstance(processed, dict):
        for name in possible_names:
            if name in processed:
                return processed[name]
    return default


summary_rows = []
processed_tickets = []

for ticket_id, ticket in enumerate(tickets):
    processed = process_one_ticket(ticket)
    processed_tickets.append(processed)

    tokens = get_field(processed, ["tokens", "all_tokens", "token_list"], [])
    punctuation = get_field(
        processed, ["punctuation_tokens", "punctuation", "punct_tokens"], []
    )
    stopword_tokens = get_field(
        processed, ["stopwords", "stopword_tokens", "stop_words"], []
    )
    final_tokens = get_field(
        processed, ["final_tokens", "final", "clean_tokens"], []
    )

    # If the starter function uses different return keys, these columns may
    # need to be mapped to its actual output keys.
    summary_rows.append({
        "ticket_id": ticket_id,
        "sentences": sentence_count(ticket),
        "tokens": len(tokens),
        "punctuation_tokens": len(punctuation),
        "stopwords": len(stopword_tokens),
        "unique_tokens": len(set(str(t).lower() for t in tokens)),
        "final_tokens": len(final_tokens),
    })

summary_df = pd.DataFrame(summary_rows)
count_columns = [
    "sentences", "tokens", "punctuation_tokens", "stopwords",
    "unique_tokens", "final_tokens"
]
total_row = {"ticket_id": "TOTAL"}
for column in count_columns:
    total_row[column] = summary_df[column].sum()
summary_with_total = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

print("Q1. Preprocessing summary:")
print(summary_with_total.to_string(index=False))

if tickets:
    selected_index = 0
    selected_processed = processed_tickets[selected_index]
    print(f"\nQ1. Details for ticket {selected_index}:")
    print("Raw ticket:", tickets[selected_index])
    print("Tokens:", get_field(selected_processed, ["tokens", "all_tokens", "token_list"], []))
    print("Punctuation tokens:", get_field(
        selected_processed, ["punctuation_tokens", "punctuation", "punct_tokens"], []
    ))
    print("Stopwords:", get_field(
        selected_processed, ["stopwords", "stopword_tokens", "stop_words"], []
    ))
    print("Final tokens:", get_field(
        selected_processed, ["final_tokens", "final", "clean_tokens"], []
    ))


# Q2. Regex extraction and replacement:
# Use re.findall() to extract long words, numbers/versions, capitalized words,
# and vowel-starting words. Use re.sub() to mask email addresses, URLs, phones.
# The patterns are applied to raw tickets and TEST_LINES.

TEST_LINES = globals().get("TEST_LINES", [
    "Please email helpdesk@example.com about version 2.0.1.",
    "The Wi-Fi router shows 95.5% packet loss; call +1-800-555-0199.",
    "Alex reported that https://support.example.com/login is not working.",
])

email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
url_pattern = re.compile(r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+", re.IGNORECASE)
phone_pattern = re.compile(r"(?<!\w)(?:\+?\d[\d().\-\s]{7,}\d)(?!\w)")
long_word_pattern = re.compile(r"\b[A-Za-z]{6,}\b")
number_pattern = re.compile(r"\b\d+(?:\.\d+)*\b")
capitalized_pattern = re.compile(r"\b[A-Z][a-zA-Z]*\b")
vowel_word_pattern = re.compile(r"\b[AEIOUaeiou][A-Za-z]*\b")


def redact_contact_info(text):
    text = email_pattern.sub("<EMAIL>", text)
    text = url_pattern.sub("<URL>", text)
    text = phone_pattern.sub("<PHONE>", text)
    return text


print("\nQ2. Regex extraction and replacement:")
for line_number, line in enumerate(list(tickets) + list(TEST_LINES), start=1):
    print(f"\nLine {line_number}: {line}")
    print("Words longer than 5 letters:", long_word_pattern.findall(line))
    print("Numbers/versions:", number_pattern.findall(line))
    print("Capitalized words:", capitalized_pattern.findall(line))
    print("Words beginning with a vowel:", vowel_word_pattern.findall(line))
    print("Redacted:", redact_contact_info(line))


# Q3. Custom tokenization:
# Implement my_tokenize(text) with one RegexpTokenizer pattern preserving
# contractions, hyphenated words, decimals, versions, emails, URLs, and phones.
# Run it on TEST_LINES and tickets; compare with word_tokenize().

custom_pattern = (
    r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+|"  # URLs
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"  # email
    r"\+?\d[\d().\-\s]{7,}\d|"  # phone numbers
    r"\d+(?:\.\d+)+|"  # decimals and versions
    r"[A-Za-z]+(?:-[A-Za-z]+)*(?:['’][A-Za-z]+)?|"  # words, hyphens, contractions
    r"[^\w\s]"  # punctuation
)
custom_tokenizer = RegexpTokenizer(custom_pattern)


def my_tokenize(text):
    return custom_tokenizer.tokenize(text)


print("\nQ3. Custom tokenizer and word_tokenize comparison:")
for line in list(TEST_LINES) + list(tickets):
    try:
        standard_tokens = word_tokenize(line)
    except LookupError:
        nltk.download("punkt", quiet=True)
        standard_tokens = word_tokenize(line)
    custom_tokens = my_tokenize(line)
    print("\nText:", line)
    print("my_tokenize:", custom_tokens)
    print("word_tokenize:", standard_tokens)

print("\nQ3. Typical differences:")
print("1. The custom tokenizer keeps hyphenated terms such as Wi-Fi together; "
      "word_tokenize() may split them around the hyphen.")
print("2. The custom tokenizer keeps contractions such as isn't together; "
      "word_tokenize() commonly splits the contraction into multiple tokens.")
print("3. The custom tokenizer is designed to keep URLs, emails, phone numbers, "
      "and dotted versions such as 2.0.1 as single tokens, while word_tokenize() "
      "may split these into punctuation and word/number pieces.")


# Q4. Stemming and lemmatization:
# On unique final tokens, compare Porter, Lancaster, Regexp Stemmer, and
# WordNet Lemmatizer using POS information from nltk.pos_tag().
# The regex suffix is selected as "ing", which can be tested against words
# such as "working" and "connecting" if they occur in the personalized tickets.

all_final_tokens = []
for processed in processed_tickets:
    all_final_tokens.extend(
        get_field(processed, ["final_tokens", "final", "clean_tokens"], [])
    )

unique_final_tokens = sorted(set(str(token).lower() for token in all_final_tokens))
porter = PorterStemmer()
lancaster = LancasterStemmer()
regexp_stemmer = RegexpStemmer(r"ing$", min=4)
lemmatizer = WordNetLemmatizer()


def wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return "a"
    if treebank_tag.startswith("V"):
        return "v"
    if treebank_tag.startswith("N"):
        return "n"
    if treebank_tag.startswith("R"):
        return "r"
    return "n"


pos_tags = pos_tag(unique_final_tokens) if unique_final_tokens else []
pos_by_word = {word: tag for word, tag in pos_tags}

stem_rows = []
for word in unique_final_tokens:
    tag = pos_by_word.get(word, "NN")
    row = {
        "word": word,
        "Porter": porter.stem(word),
        "Lancaster": lancaster.stem(word),
        "Regexp": regexp_stemmer.stem(word),
        "WordNet_lemma": lemmatizer.lemmatize(word, wordnet_pos(tag)),
    }
    if len(set(row.values())) > 2:
        stem_rows.append(row)

stem_comparison_df = pd.DataFrame(stem_rows)
print("\nQ4. Words for which stemming/lemmatization outputs differ:")
if stem_comparison_df.empty:
    print("No differing rows were found in the available final tokens.")
else:
    print(stem_comparison_df.to_string(index=False))

print("\nQ4. Regexp stemmer choice: the suffix 'ing$' removes a common present-participle "
      "ending. Check that at least two words in the generated tickets actually end "
      "in 'ing' (for example, 'working' or 'connecting') before describing them as "
      "dataset examples. The suffix is intentionally simple and can produce "
      "undesirable forms.")


# Q5. Corpus statistics:
# S1 = all tokens, S2 = tokens without punctuation, S3 = final tokens,
# S4 = S3 after Porter stemming. Calculate total tokens, unique tokens and TTR.
# Print top 10 words of S1/S3, plot log(rank) vs log(frequency) for S3, and
# print top 5 bigrams of S1/S3.

def flatten_token_field(processed_list, names):
    output = []
    for item in processed_list:
        output.extend(get_field(item, names, []))
    return [str(token).lower() for token in output]


S1 = flatten_token_field(processed_tickets, ["tokens", "all_tokens", "token_list"])
S3 = flatten_token_field(processed_tickets, ["final_tokens", "final", "clean_tokens"])

punctuation_all = set()
for processed in processed_tickets:
    punctuation_all.update(
        str(token).lower() for token in get_field(
            processed, ["punctuation_tokens", "punctuation", "punct_tokens"], []
        )
    )
S2 = [token for token in S1 if token not in punctuation_all]
S4 = [porter.stem(token) for token in S3]


def corpus_stats(name, tokens):
    total = len(tokens)
    unique = len(set(tokens))
    ttr = unique / total if total else 0
    return {"Corpus": name, "Total tokens": total, "Unique tokens": unique, "TTR": round(ttr, 4)}


stats_df = pd.DataFrame([
    corpus_stats("S1: all tokens", S1),
    corpus_stats("S2: without punctuation", S2),
    corpus_stats("S3: final tokens", S3),
    corpus_stats("S4: Porter-stemmed S3", S4),
])
print("\nQ5. Corpus statistics:")
print(stats_df.to_string(index=False))

print("\nQ5. Top 10 words in S1:")
print(Counter(S1).most_common(10))
print("\nQ5. Top 10 words in S3:")
print(Counter(S3).most_common(10))


def top_bigrams(tokens, count=5):
    bigrams = zip(tokens, tokens[1:])
    return Counter(bigrams).most_common(count)


print("\nQ5. Top 5 bigrams in S1:")
print(top_bigrams(S1))
print("\nQ5. Top 5 bigrams in S3:")
print(top_bigrams(S3))

frequencies = Counter(S3)
ranked = sorted(frequencies.values(), reverse=True)
if ranked:
    ranks = list(range(1, len(ranked) + 1))
    plt.figure()
    plt.plot(np.log(ranks), np.log(ranked), marker="o", linestyle="none")
    plt.xlabel("log(rank)")
    plt.ylabel("log(frequency)")
    plt.title("Q5. Log(rank) vs log(frequency) for S3")
    plt.tight_layout()
    plt.savefig("assignment7_s3_rank_frequency.png", dpi=150)
    plt.show()


# Q6. Interpretation:
# (a) Why does TTR change from S1 to S4?
# (b) Give one tokenizer difference.
# (c) Give an undesirable stemming example.
# (d) Give an example where POS-aware lemmatization preserves meaning better.

print("\nQ6(a). TTR usually changes because preprocessing removes punctuation/stopwords "
      "and stemming maps related word forms to a shared stem. Removing frequent "
      "tokens can raise TTR, while stemming usually reduces the number of unique "
      "forms and can lower TTR. The actual direction depends on this ticket dataset.")
print("Q6(b). Example: my_tokenize('isn\\'t working') keeps 'isn't' as one token, "
      "while word_tokenize() commonly separates it into 'is' and \"n't\".")
print("Q6(c). Example: a simple suffix rule can turn 'testing' into 'test', which "
      "may be acceptable, but blindly removing a suffix can also create a non-word "
      "or change meaning. Use an actual word from the generated tickets when writing "
      "the final report.")
print("Q6(d). Example: POS-aware lemmatization can map 'was' to 'be' when tagged as "
      "a verb, while a stemmer may return a fragment such as 'wa'. Use a word and "
      "POS tag present in your generated ticket data if your instructor expects "
      "dataset-specific examples.")
