from typing import Dict, Final

class RuleBasedChatbot:
    # Defining core configuration constants to eliminate magic strings
    KILL_COMMAND: Final[str] = "exit"
    FALLBACK_RESPONSE: Final[str] = "I do not understand. Please try another query or command."
    
    def __init__(self) -> None:
        """
        Initializes the Knowledge Base using an optimized hash map (dictionary).
        Provides direct access with O(1) constant time complexity.
        """
        self._knowledge_base: Dict[str, str] = {
            "hello": "Hi there! Welcome to the DecodeLabs intelligent interface.",
            "hi": "Hello! How can I assist you with your AI engineering track today?",
            "help": "I can help you understand rule-based logic, dictionaries, and continuous control loops.",
            "status": "System Status: Online. All deterministic logic engines are operating within normal parameters.",
            "project": "Project 1 is the essential foundation phase: building a Rule-Based AI Chatbot.",
            "decode labs": "DecodeLabs is where you master the art of precision logic before entering probability engines.",
            "bye": "Goodbye! Keep experimenting and building your portfolio."
        }

    def _sanitize_input(self, raw_text: str) -> str:
        """
        PHASE 1: INPUT & SANITIZATION
        Converts raw text feed to lowercase and strips padding whitespace.
        """
        return raw_text.lower().strip()

    def _process_intent(self, clean_input: str) -> str:
        """
        PHASE 2: PROCESS (The Logic Skeleton)
        Executes an atomic lookup with fallback handling in a single instruction.
        """
        return self._knowledge_base.get(clean_input, self.FALLBACK_RESPONSE)

    def start_loop(self) -> None:
        """
        THE HEARTBEAT: Starts the continuous digital control loop.
        The system processes inputs deterministically until an exit sequence is issued.
        """
        print("====================================================")
        print("     DECODELABS DETERMINISTIC LOGIC ENGINE v1.1     ")
        print("====================================================")
        print(f"Type your message below. Enter '{self.KILL_COMMAND}' to terminate.\n")

        while True:
            try:
                # 1. INPUT (Raw Feed)
                user_raw = input("You: ")
                
                # 2. PROCESS (Sanitization & Normalization)
                clean_input = self._sanitize_input(user_raw)
                
                # Exit Strategy Evaluation
                if clean_input == self.KILL_COMMAND:
                    print("Bot: Terminating continuous loop. Process killed successfully. Goodbye!")
                    break
                    
                # Intent Evaluation
                reply = self._process_intent(clean_input)
                
                # 3. OUTPUT (Feedback Loop)
                print(f"Bot: {reply}\n")
                
            except (KeyboardInterrupt, EOFError):
                print("\nBot: Process interrupted externally. Cleaning up state safely.")
                break


if __name__ == "__main__":
    # Instantiate and spin up the engine
    bot = RuleBasedChatbot()
    bot.start_loop()