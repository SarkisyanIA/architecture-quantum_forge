import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Загружаем документы с метаданными
documents = []
data_folder = "./knowledge_base"

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as f:
            content = f.read()
            # Создаем объект Document, который хранит текст и "шапку" с данными
            doc = Document(
                page_content=content,
                metadata={"source": filename, "source_path": os.path.abspath(filename)}
            )
            documents.append(doc)

# 2. Настраиваем сплиттер
# chunk_size = 500 символов (~100-150 слов для русского языка)
# chunk_overlap = 50 символов (нужен, чтобы смысл не терялся на стыках)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True,  # Покажет позицию чанка в исходном тексте
)

# 3. Делим документы (метаданные сохранятся автоматически!)
chunks = text_splitter.split_documents(documents)

# 4. Смотрим результат
print(f"📄 Загружено документов: {len(documents)}")
print(f"✂️  Создано чанков: {len(chunks)}")
for chunk in chunks[:79]:
    print("---")
    print(f"Источник: {chunk.metadata['source']}")
    print(f"Позиция (символ): {chunk.metadata.get('start_index', 'N/A')}")
    print(f"Текст: {chunk.page_content[:100]}...")