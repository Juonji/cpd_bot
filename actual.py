import logging
import telebot
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Включаем ведение журнала
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Токен, который вы получили от @BotFather
def token_loader():
    try:
        with open("token.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error("Файл с токеном не найден!")
        exit(0)

# Загрузка инструкций и настроек
def prompt_loader():
    try:
        with open("prompt.yaml", "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error("Файл с prompt не найден!")
        exit(0)

THEMATIC_INSTRUCTION = prompt_loader()

# Авторизация в GigaChat
llm = GigaChat(
    credentials="ZjRhZjBkMWMtNzVmZC00ZDFhLWI5ODUtNmYzZTgyOGUyMDg0OjgzMzA5ZjNjLTcxZjctNDhlNC1hZjhhLWMwZDUwMjQ3NjA3Nw==",
    scope="GIGACHAT_API_PERS",
    model="GigaChat",
    verify_ssl_certs=False,
    streaming=False,
)

# Инициализация бота
bot = telebot.TeleBot(token_loader())

# Память и граф для управления состоянием пользователей
workflow = StateGraph(state_schema=MessagesState)

# Функция вызова модели
def call_model(state: MessagesState):
    try:
        response = llm.invoke(state["messages"])
        return {"messages": response}
    except Exception as e:
        logging.error(f"Ошибка в работе GigaChat: {e}")
        return {"messages": [SystemMessage(content="Произошла ошибка. Попробуйте позже.")]}

# Добавление узла и перехода в граф
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Персистенция для сохранения контекста
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Обработка команды /start
@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    thread_id = str(user_id)  # Уникальный thread_id для каждого пользователя
    config = {"configurable": {"thread_id": thread_id}}
    
    # Сбрасываем историю для нового пользователя
    input_messages = [SystemMessage(content=THEMATIC_INSTRUCTION)]
    app.invoke({"messages": input_messages}, config)
    
    bot.reply_to(message, "Привет! Я готов помочь. Начнем!")

# Обработка сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id
    thread_id = str(user_id)  # Уникальный thread_id для каждого пользователя
    
    config = {"configurable": {"thread_id": thread_id}}
    user_message = HumanMessage(content=message.text)

    try:
        # Добавляем новое сообщение пользователя
        output = app.invoke({"messages": [user_message]}, config)
        response_message = output["messages"][-1].content  # Последний ответ модели
        
        # Отправляем ответ пользователю
        bot.reply_to(message, response_message)
    except Exception as e:
        logging.error(f"Ошибка в работе GigaChat: {e}")
        bot.reply_to(message, "Ошибка. Попробуйте позже.")

# Основная функция
if __name__ == '__main__':
    bot.polling(none_stop=True)
