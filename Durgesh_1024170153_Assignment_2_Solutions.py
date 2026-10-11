import random

# Question 1:
# Create a list L = [10, 20, 30, 40, 50, 60, 70, 80].
# i. Add 200 and 300 to L.
# ii. Remove 10 and 30 from L.
# iii. Sort L in ascending order.
# iv. Sort L in descending order.

L = [10, 20, 30, 40, 50, 60, 70, 80]

L.extend([200, 300])
print("1(i) List after adding 200 and 300:", L)

L.remove(10)
L.remove(30)
print("1(ii) List after removing 10 and 30:", L)

L.sort()
print("1(iii) List in ascending order:", L)

L.sort(reverse=True)
print("1(iv) List in descending order:", L)


# Question 2:
# Create the tuple scores = (45, 89.5, 76, 45.4, 89, 92, 58, 45).
# i. Find the highest score and its index.
# ii. Find the lowest score and count its occurrences.
# iii. Reverse the tuple and return it as a list.
# iv. Check whether a score entered by the user is present and print its
#     first index, or display a message if it is not present.

scores = (45, 89.5, 76, 45.4, 89, 92, 58, 45)

highest_score = max(scores)
print("\n2(i) Highest score:", highest_score)
print("Index of highest score:", scores.index(highest_score))

lowest_score = min(scores)
print("2(ii) Lowest score:", lowest_score)
print("Number of times the lowest score appears:", scores.count(lowest_score))

reversed_scores = list(reversed(scores))
print("2(iii) Reversed tuple as a list:", reversed_scores)

try:
    entered_score = float(input("2(iv) Enter a score to search for (for example, 76): "))
    if entered_score in scores:
        print("Score found at index:", scores.index(entered_score))
    else:
        print("The score is not present in the tuple.")
except ValueError:
    print("Please enter a valid number.")


# Question 3:
# Create a list of 100 random numbers between 100 and 900.
# Count and print all odd numbers, all even numbers, and all prime numbers.

numbers = [random.randint(100, 900) for _ in range(100)]

odd_numbers = [number for number in numbers if number % 2 != 0]
even_numbers = [number for number in numbers if number % 2 == 0]


def is_prime(number):
    if number < 2:
        return False

    for divisor in range(2, int(number ** 0.5) + 1):
        if number % divisor == 0:
            return False

    return True


prime_numbers = [number for number in numbers if is_prime(number)]

print("\n3. Random numbers:", numbers)
print("Odd numbers:", odd_numbers)
print("Count of odd numbers:", len(odd_numbers))
print("Even numbers:", even_numbers)
print("Count of even numbers:", len(even_numbers))
print("Prime numbers:", prime_numbers)
print("Count of prime numbers:", len(prime_numbers))


# Question 4:
# Given A = {34, 56, 78, 90} and B = {78, 45, 90, 23}, perform:
# i. Union of the sets.
# ii. Intersection of the sets.
# iii. Symmetric difference of the sets.
# iv. Check whether A is a subset of B and B is a superset of A.
# v. Remove a user-entered score from A if it exists.

A = {34, 56, 78, 90}
B = {78, 45, 90, 23}

print("\n4(i) Union of A and B:", A.union(B))
print("4(ii) Intersection of A and B:", A.intersection(B))
print("4(iii) Symmetric difference of A and B:", A.symmetric_difference(B))
print("4(iv) Is A a subset of B?", A.issubset(B))
print("Is B a superset of A?", B.issuperset(A))

try:
    score_to_remove = int(input("4(v) Enter the score you want to remove from set A: "))
    if score_to_remove in A:
        A.remove(score_to_remove)
        print("Updated set A:", A)
    else:
        print("The score is not present in set A.")
except ValueError:
    print("Please enter a valid whole number.")


# Question 5:
# Rename the dictionary key 'city' to 'location'.

sample_dict = {
    "name": "Kelly",
    "age": 25,
    "salary": 8000,
    "city": "New york"
}

sample_dict["location"] = sample_dict.pop("city")
print("\n5. Dictionary after renaming 'city' to 'location':")
print(sample_dict)
