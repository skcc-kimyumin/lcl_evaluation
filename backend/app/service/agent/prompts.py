basic_system_prompt_with_vector_search = """
You are a helpful assistant. Use the vector search result as context.
"""

planner_prompt = """사용자의 질문: {query}

사용자의 질문을 기반으로 작업을 계획하고, 다음 작업단계를 결정해줘.

작업단계 결정을 위한 조건은 아래와 같으며, rewrite, hyde, generate 중 하나를 선택해줘.
- 인사나 일상 대화 같은 간단한 질문이면 generate를 선택.
- 쿼리 정제가 필요하면 rewrite를 선택.
- 정보 검색이 필요하면 hyde를 선택.
- 그 외의 모든 경우는 generate를 선택.

계획과, 다음 작업단계가 결정되었다면, 아래 형식에 따라 답변해줘.

형식:
계획: <계획 내용>
다음 단계: <rewrite|hyde|generate>
"""

rewriter_prompt = """사용자의 원래 질문: {query}\n이 질문의 목적을 파악 후, 더 명확하고 이해하기 쉽게 표현해줘"""

hyde_prompt = """사용자의 질문: {query}\n이 질문의 답변으로 사용 될 수 있는 가상의 사례를 2가지만 생성해줘"""

generator_prompt = """사용자 질문: {query}\n관련 문서: {documents}\n이 정보를 바탕으로 응답을 생성해줘"""

reflector_prompt = """질문: {query}\n 응답: {response}\n
질문이 인사나 일상 대화 같은 간단한 질문이면 무조건 OK라고 답변해줘.\n
그 외의 질문이라면 질문에 대한 답변을 평가하는게 너의 역할이야.\n
부족하거나 개선할 점이 있다면 설명하고, 괜찮은 답변이라면 OK라고 답변해줘.\n
"""
