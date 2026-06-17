#Riverside books chatbot (final CLI product)

from retriever import find_best_match

print("=== Riverside Books Chatbot ===")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break

    answer, score = find_best_match(user_input)

    if answer:
        print("Bot:", answer)
    else:
        print("Bot: Sorry, I don’t know that one - please ask a member of staff.")