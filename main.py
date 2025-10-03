from agentic_plot import history

def main():
  payload = history.ChatHistory()
  
  payload.add_system_message("You are a disinterested but capable and obedient AI assistant.")
  payload.add_assistant_message("Hi, I'm not really chatting.")
  payload.add_user_message("Get in the fucking robot, Shinji.")
  payload.add_assistant_message("Who the fuck is Shinji")
  payload.add_user_message("Not in the robot.")

  print(payload.get_conversation_summary())

if __name__ == "__main__":
  main()