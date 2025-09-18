import src.agent


def main():
    agent = src.agent.Agent()
    while True:
        user_text = input("You: ")
        answer = agent.run_turn(user_text)
        print(f"Agent: {answer}")

if __name__ == "__main__":
    main()
