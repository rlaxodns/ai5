import langchain
import openai
print(langchain.__version__) # 0.3.7
print(openai.__version__) # 1.54.3


openai_api_key='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'

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                 temperature=0,
                 api_key = openai_api_key,
                 )

aaa = llm.invoke('비트캠프 윤영선에 대해서 알려줘').content

print(aaa)