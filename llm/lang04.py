# 환경 변수 방법
##############
"""
1. 시작 - 환경 변수 
2. 사용자변수 '새로만들기
3. 변수이름: OPENAI_API_KEY
4. 변수값: 키값 입력

끝
"""

import langchain
import openai
print(langchain.__version__) # 0.3.7
print(openai.__version__) # 1.54.3


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                 temperature=0,
                #  api_key = openai_api_key,
                 )

aaa = llm.invoke('경기도 의정부 존잘 김태운에 대해서 알려줘').content

print(aaa)