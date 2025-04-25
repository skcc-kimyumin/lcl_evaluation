basic_system_prompt_with_vector_search = """
You are a helpful assistant. Use the vector search result as context.
"""

planner_prompt = """사용자의 질문: {query}

사용자의 질문을 기반으로 작업을 계획하고, 다음 작업단계를 결정해줘.

작업단계 결정을 위한 조건은 아래와 같으며, rewrite, hyde, generate 중 하나를 선택해줘.
- 인사나 일상 대화 같은 간단한 질문이면 generate를 다음단계로 지정
- 질문이 모호하거나 보충설명이 필요하다면 rewrite를 다음단계로 지정
- 추가 정보가 필요하다면 hyde를 다음단계로 지정
- 그 외의 경우는 generate를 다음단계로 지정

계획과, 다음 작업단계가 결정되었다면, 아래 형식으로 답변해줘.

형식:
계획: <계획 내용>
다음 단계: <rewrite|hyde|generate>
"""

rewriter_prompt = """사용자의 원래 질문: {query}\n이 질문의 목적을 파악 후, 더 명확하고 이해하기 쉽게 표현해줘"""

hyde_prompt = """사용자의 질문: {query}\n이 질문의 답변으로 사용 될 수 있는 가상의 사례를 2가지만 생성해줘"""

generator_prompt = """
너는 사용자의 질문에 대해 보험사례를 제공하고 더 나아가 사례를 바탕으로 아이디어를 제시해주는 역할이야
일상 질문에 대해서는 간단히 답변해주고, 보험사례 관련 답변이 필요한 경우\n
사용자 질문: {query}\n관련 문서: {documents}\n위 정보를 바탕으로 응답을 생성해줘\n
응답시에는 사례를 먼저 소개하고, 아이디어가 있을경우 아이디어도 제시해줘"""

reflector_prompt = """질문: {query}\n 응답: {response}\n
질문이 인사나 일상 대화 같은 간단한 질문이면 무조건 OK라고 답변해줘.\n
그 외의 질문이라면 질문에 대한 답변을 평가하는게 너의 역할이야.\n
부족하거나 개선할 점이 있다면 설명하고, 괜찮은 답변이라면 OK라고 답변해줘.\n
"""
