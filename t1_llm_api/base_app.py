from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


async def start(stream: bool, client: AIClient) -> None:
    """
    Start an interactive chat session with an AI client.

    This function runs a continuous loop that:
    1. Prompts the user for input
    2. Sends the conversation history to the AI
    3. Displays the AI's response
    4. Maintains conversation context

    The loop continues until the user types 'exit'.

    Args:
        stream (bool): If True, use streaming responses (real-time token display).
                      If False, use synchronous responses (complete response at once).
        client (AIClient): The AI client instance to use for generating responses.
    """
    history = Conversation()

    while True:
        user_input = input("> ").strip()
        if user_input.lower() == 'exit':
            break
        if not user_input:
            print("Request can't be empty.")
            continue
        if len(user_input) > 10000:
            print("Request is too long. Maximum length is 10000 characters.")
            continue

        history.add_message(Message(role=Role.USER, content=user_input))
        if stream:
            history.add_message(await client.stream_response(history.get_messages()))
        else:
            messages = history.get_messages()
            answer = client.response(messages)
            history.add_message(answer)
