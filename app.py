# 랭체인에서 쳇모델을 가져오는데 openaidml gpt-4o-mini 모델 사용
# 환경설정을 .env에 저장해두고 그 설정을 읽어들인다. (pip install dotenv)
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("Hi!"),
]

# response = model.invoke(messages)
# print(response)
# print(model.invoke(messages))

# for token in model.stream(messages):
#     print(token.content, end="|")

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")] # 동적 입력 대상 변수
)

prompt = prompt_template.invoke({"language": "Korean", "text": "hi!"})

# print(prompt)

response = model.invoke(prompt)
print(response.content)