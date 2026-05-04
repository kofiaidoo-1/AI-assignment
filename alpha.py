from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: FAQ Data
questions = [
    "What is AI?",
    "What is machine learning?",
    "What are programming languages?",
    "How do computers work?",
    "What is API ?"
]

answers = [
    "AI is a field of computer science focused on building machines and software capable of simulating human intelligence.",
    "Machine learning is a subfield of Artificial intelligence that allows computers to learn from data and improve their performance without being explicity programmed for every task .",
    "programming languages are structured sets of instructions and symbols used to communicate with computers.",
    "Computers work by processing binary data through a cycle of input, processing, storage, and output .",
    "An API is a set of rules that allows two different pieces of software to talk to each other."
]

# Step 2: Train the model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Step 3: Chatbot function
def chatbot(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()

    # If no good match
    if similarity[0][index] < 0.3:
        return "Sorry, I don't understand."

    return answers[index]

# Step 4: Run chatbot
print("Chatbot is ready! Type 'exit' to stop.\n")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = chatbot(user)
    print("Bot:", response)