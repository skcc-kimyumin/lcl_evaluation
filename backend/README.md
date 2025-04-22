# Project Template (FastAPI)

이 프로젝트는 **FastAPI** 기반의 백엔드 템플릿입니다.  
기본적인 API 구조와 데이터베이스, 로깅, 에러 핸들링, VectorDB 연동 등의 기능을 포함하고 있습니다.  

## 디렉터리 구조

```sh
.
└── app
    ├── api                     # API endpoint 코드
    │   └── routes
    ├── core                    # 공통적으로 사용되는 코드
    ├── database                # DB 연결, 모델, CRUD 등의 코드
    │   ├── model
    │   └── repository
    ├── error                   # 에러 핸들링 코드
    ├── log                     # 로깅 설정 코드
    ├── service                 # 기능 구현을 위한 서비스 코드
    │    └── model
    ├── vectordb                # VectorDB 실행을 위한 코드 
```

## 개발자 환경 설정

### 1. Conda 설치
FastAPI 프로젝트를 실행하기 위해 먼저 Conda를 설치해야 합니다. Conda가 설치되지 않은 경우 [공식 사이트](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation)에서 Miniconda 또는 Anaconda를 다운로드하여 설치하세요.

### 2. VSCode에서 가상환경 생성
터미널을 열고 아래 명령어를 실행하여 새로운 Conda 가상환경을 생성합니다.

```sh
conda create -n edu-template python=3.10
```

이후 가상환경을 활성화합니다.

```sh
conda activate edu-template
```

### 3. 해당 가상환경으로 터미널 로딩
VSCode에서 Conda 가상환경을 사용하려면, 아래 절차를 따릅니다.

1. `Ctrl + Shift + P`를 눌러 "Python: Select Interpreter"를 검색합니다.
2. 생성한 가상환경(`edu-template`)을 선택합니다.
3. 터미널을 새로 열어 해당 환경이 활성화되었는지 확인합니다.


### 4. 아래 명령어를 실행하여 requirements.txt 파일에 정의된 패키지를 설치합니다.

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. VSCode에서 Task 실행
VSCode에서 특정 작업을 자동화하기 위해 `tasks.json`을 사용합니다. 아래 절차를 통해 task를 실행할 수 있습니다.

1. `Ctrl + Shift + P`를 눌러 "Tasks: Run Task"를 검색합니다.
2. 원하는 Task를 선택하여 실행합니다.
3. `tasks.json` 파일을 참고하여 적절한 Task를 설정할 수 있습니다.


### 6. Debug로 Backend 실행
FastAPI 백엔드를 디버깅하려면 `launch.json` 파일을 활용하여 실행할 수 있습니다.

1. VSCode에서 `F5`를 눌러 Debug 모드로 실행합니다.
2. `launch.json`에 설정된 대로 FastAPI 서버가 실행됩니다.
3. 필요에 따라 Breakpoint를 설정하여 코드 흐름을 분석할 수 있습니다.

---

실행이 완료되면 아래 주소로 접속하여 확인합니다.  
http://localhost:8000/docs#/

