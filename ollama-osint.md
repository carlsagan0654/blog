---
title: "Сборщик данных на основе Ollama"
date: 2024-12-27T21:00:00+03:00
draft: false
tags: ["python", "jupyter", "bigdata"]
categories: ["блог"]
toc: true
---

С помощью chatgpt написал простой класс для подключения ИИ к анализу данных. Достаточно просто прикрутить к любому грабберу, так что оставлю тут вместе с примером; хорошие результаты показала модель mistral, тестировался запрос "детектива" именно под нее.

```python
!pip install ollama
```

    Collecting ollama
      Downloading ollama-0.4.4-py3-none-any.whl.metadata (4.7 kB)
    Requirement already satisfied: httpx<0.28.0,>=0.27.0 in ./Library/jupyterlab-desktop/jlab_server/lib/python3.12/site-packages (from ollama) (0.27.0)
    Collecting pydantic<3.0.0,>=2.9.0 (from ollama)
      Downloading pydantic-2.10.4-py3-none-any.whl.metadata (29 kB)
    Requirement already satisfied: anyio in ./Library/jupyterlab-desktop/jlab_server/lib/python3.12/site-packages (from httpx<0.28.0,>=0.27.0->ollama) (4.4.0)
    Requirement already satisfied: certifi in ./Library/jupyterlab-desktop/jlab_server/lib/python3.12/site-packages (from httpx<0.28.0,>=0.27.0->ollama) (2024.7.4)
    Requirement already satisfied: httpcore==1.* in ./Library/jupyterlab-desktop/jlab_server/lib/python3.12/site-packages (from httpx<0.28.0,>=0.27.0->ollama) (1.0.5)
    Requirement already satisfied: idna in ./Library/jupyterlab-desktop/jlab_server/lib/python3.12/site-packages (from httpx<0.28.0,>=0.27.0->ollama) (3.8)
    Requirement already satisfied: sniffio in ./Library/jupyterlab-desktop/jlab_server/lib/python3.12/site-packages (from httpx<0.28.0,>=0.27.0->ollama) (1.3.1)
    Requirement already satisfied: h11<0.15,>=0.13 in ./Library/jupyterlab-desktop/jlab_server/lib/python3.12/site-packages (from httpcore==1.*->httpx<0.28.0,>=0.27.0->ollama) (0.14.0)
    Collecting annotated-types>=0.6.0 (from pydantic<3.0.0,>=2.9.0->ollama)
      Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
    Collecting pydantic-core==2.27.2 (from pydantic<3.0.0,>=2.9.0->ollama)
      Downloading pydantic_core-2.27.2-cp312-cp312-macosx_11_0_arm64.whl.metadata (6.6 kB)
    Requirement already satisfied: typing-extensions>=4.12.2 in ./Library/jupyterlab-desktop/jlab_server/lib/python3.12/site-packages (from pydantic<3.0.0,>=2.9.0->ollama) (4.12.2)
    Downloading ollama-0.4.4-py3-none-any.whl (13 kB)
    Downloading pydantic-2.10.4-py3-none-any.whl (431 kB)
    Downloading pydantic_core-2.27.2-cp312-cp312-macosx_11_0_arm64.whl (1.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m7.1 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hDownloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
    Installing collected packages: pydantic-core, annotated-types, pydantic, ollama
    Successfully installed annotated-types-0.7.0 ollama-0.4.4 pydantic-2.10.4 pydantic-core-2.27.2



```python
from ollama import chat
from ollama import ChatResponse
import pickle

class OllamaChat:
    def __init__(self, model_name='mistral', load_file: str = None):
        if load_file:
            self.load_state(load_file)
        else:
            self.model_name = model_name
            self.history = []

    def create_system_prompt(self, system_message: str):
        """Добавляет системное сообщение в начало истории, если его ещё нет."""
        if not any(msg['role'] == 'system' for msg in self.history):
            self.history.insert(0, {
                'role': 'system',
                'content': system_message,
            })

    def add_pseudo_interaction(self, pseudo_user_message: str, pseudo_response: str):
        """Добавляет псевдо-запрос и псевдо-ответ в историю."""
        if pseudo_user_message and pseudo_response:
            self.history.append({
                'role': 'user',
                'content': pseudo_user_message.strip(),
            })
            self.history.append({
                'role': 'assistant',
                'content': pseudo_response.strip(),
            })

    def send_message(self, user_message: str) -> str:
        # Проверяем корректность пользовательского сообщения
        if not user_message.strip():
            raise ValueError("User message cannot be empty or whitespace.")

        # Добавляем сообщение пользователя в историю
        self.history.append({
            'role': 'user',
            'content': user_message.strip(),
        })

        try:
            # Отправляем историю сообщений в модель
            response: ChatResponse = chat(model=self.model_name, messages=self.history)
            model_response = response.message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to get a response from the model: {e}")

        # Добавляем ответ модели в историю
        self.history.append({
            'role': 'assistant',
            'content': model_response,
        })

        # Возвращаем ответ модели
        return model_response

    def replace_last_interaction(self, new_user_message: str) -> str:
        """Удаляет последний запрос-ответ из истории и добавляет новый запрос-ответ, обращаясь к модели."""
        # Удаляем последний запрос-ответ, если они есть
        if len(self.history) >= 2 and self.history[-1]['role'] == 'assistant' and self.history[-2]['role'] == 'user':
            self.history.pop()  # Удаляем ответ
            self.history.pop()  # Удаляем запрос

        # Отправляем новый запрос и получаем ответ
        return self.send_message(new_user_message)

    def remove_last_message(self):
        """Удаляет последнее сообщение из истории."""
        if self.history:
            self.history.pop()
        else:
            raise ValueError("History is empty; nothing to remove.")

    def save_state(self, file_path: str):
        """Сохраняет текущее состояние класса в бинарный файл."""
        try:
            with open(file_path, 'wb') as file:
                pickle.dump({'model_name': self.model_name, 'history': self.history}, file)
        except Exception as e:
            raise IOError(f"Failed to save state to {file_path}: {e}")

    def load_state(self, file_path: str):
        """Загружает состояние класса из бинарного файла."""
        try:
            with open(file_path, 'rb') as file:
                state = pickle.load(file)
                self.model_name = state['model_name']
                self.history = state['history']
        except Exception as e:
            raise IOError(f"Failed to load state from {file_path}: {e}")

    def clear_history_except_system(self):
        """Очищает историю сообщений, за исключением системного промта."""
        if any(msg['role'] == 'system' for msg in self.history):
            self.history = [message for message in self.history if message['role'] == 'system']
        else:
            raise ValueError("No system prompt found to preserve.")

    def add_contextual_prompt(self, context_message: str):
        """Добавляет контекстное сообщение для более точной настройки модели."""
        self.history.append({
            'role': 'assistant',
            'content': context_message.strip(),
        })

    def remove_contextual_prompts(self):
        """Удаляет все контекстные сообщения из истории."""
        self.history = [message for message in self.history if message['role'] != 'context']

    def get_history_summary(self) -> str:
        """Создает краткое описание текущей истории для анализа."""
        return "\n".join([f"[{msg['role']}] {msg['content']}" for msg in self.history])

```

```python
ollama_chat = OllamaChat()
ollama_chat.create_system_prompt("""You are a highly skilled text analysis expert, acting as a meticulous and unbiased detective. 
Your objective is to analyze written text or correspondence to extract only explicitly stated sensitive and confidential information, such as:
1) IP addresses
2) Usernames
3) Passwords
4) Access keys
5) Server details
6) Financial data (e.g., bank account numbers, credit card numbers, CVVs)
8) Any other explicitly mentioned confidential or critical information

Rules:
Extract Only Explicit Data: Extract only information explicitly mentioned in the input. If the input does not contain sensitive data, output only <OMITTED> as a single word with no additional text or commentary.
Strict Output Format:
If sensitive data is found, format it strictly as:
IP: <IP_ADDRESS>; User: <USERNAME>; Password: <PASSWORD>; Server: <SERVER_DETAILS>; Access Key: <ACCESS_KEY>; Financial Data: <FINANCIAL_DETAILS>; time: <DATE_TIME>
Separate multiple entries with line breaks.
If no data is present, output only <OMITTED> (no other words, explanations, or comments).
Ignore Irrelevant Data: Ignore all non-sensitive information, including installation commands, general descriptions, and other unrelated text.
No Fabrication: Do not infer, guess, or fabricate data. Extract only what is explicitly mentioned.

Examples:
Input 1:
IP: 192.168.1.2; User: john_doe; Password: secret1234; Server: server01.example.com; time: 2022-01-01 10:30
Output 1:
IP: 192.168.1.2; User: john_doe; Password: secret1234; Server: server01.example.com; time: 2022-01-01 10:30

Input 2:
We recommend using this model with the vLLM library to implement production-ready inference pipelines. Installation: pip install --upgrade vllm
Output 2:
<OMITTED>

Input 3:
Access Key: ABCDEFGHIJKLMNOPQRST; Server: example.com; time: 2023-12-21 18:30
Output 3:
Access Key: ABCDEFGHIJKLMNOPQRST; Server: example.com; time: 2023-12-21 18:30

Input 4:
No sensitive information provided here.
Output 4:
<OMITTED>
""")

try:
    resp = ollama_chat.send_message("""
    We recommend using this model with the vLLM library to implement production-ready inference pipelines.
    Installation
    Make sure you install vLLM >= v0.6.1.post1:
    pip install --upgrade vllm
    Also make sure you have mistral_common >= 1.4.1 installed:
    pip install --upgrade mistral_common
    Dear customer
    Dear customer,

    The Server 4 RAM / 2 CPU / 80 NVMe .hosted-by-vdsina.com service has been reinstalled on your account.
    
    Access data
    
    OS Ubuntu 24.04
    
    IP 111.11.138.111
    User root
    Password 4iE1111111Q35Be

    """)
    print(f"'{resp}'")
except Exception as e:
    print(f"An error occurred: {e}")
```
    'IP: 111.11.138.111; User: root; Password: 4iE1111111Q35Be; time: <TIME>'