# .env 방법
##############
"""
1. 루트에 .env 파일을 만든다.
2. 파일 안에 키를 넣는다.
   .env 파일 내용
   OPENAI_API_KEY = '~~~'

3. .env가 깃에 자동으로 안올라가도록 .gitignore파일 안에 .env를 넣는다

끝
"""

import langchain
import openai
print(langchain.__version__) # 0.3.7
print(openai.__version__) # 1.54.3
print(langchain.__version__) # 0.3.7
print(openai.__version__) # 1.54.3

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                 temperature=0,
                #  api_key = openai_api_key,
                 )

aaa = llm.invoke('경기도 의정부 불주먹 김태운에 대해서 알려줘').content

print(aaa)