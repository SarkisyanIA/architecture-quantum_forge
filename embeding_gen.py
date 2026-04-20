import os
import json
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# 1. Загружаем модель для эмбеддингов
print("🔄 Загрузка модели all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Модель загружена")

# 2. Загружаем документы с метаданными
documents = []
data_folder = "./knowledge_base"

# Проверяем существование папки
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print(f"📁 Создана папка {data_folder}. Добавьте туда .txt файлы.")
    exit(1)

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Создаем объект Document, который хранит текст и "шапку" с данными
            doc = Document(
                page_content=content,
                metadata={"source": filename, "source_path": os.path.abspath(file_path)}
            )
            documents.append(doc)

if not documents:
    print("❌ В папке knowledge_base нет .txt файлов")
    exit(1)

# 3. Настраиваем сплиттер
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]  # Явно указываем разделители
)

# 4. Делим документы на чанки
print("✂️  Разбиваем документы на чанки...")
chunks = text_splitter.split_documents(documents)

# 5. Генерируем эмбеддинги для каждого чанка
print("🧮 Генерируем эмбеддинги...")
chunk_texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(chunk_texts, show_progress_bar=True)

# 6. Сохраняем результаты с эмбеддингами
print("💾 Сохраняем результаты...")
output_data = []
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    output_data.append({
        "chunk_id": i,
        "text": chunk.page_content,
        "metadata": chunk.metadata,
        "embedding": embedding.tolist(),  # конвертируем numpy array в list для JSON
        "embedding_dim": len(embedding)
    })

# Сохраняем в JSON
with open("chunks_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

# Также сохраняем эмбеддинги в формате .npy для быстрой загрузки (опционально)
embeddings_array = np.array(embeddings)
np.save("embeddings.npy", embeddings_array)

# Сохраняем метаданные отдельно для быстрого доступа
metadata_only = []
for chunk in chunks:
    metadata_only.append({
        "text": chunk.page_content,
        "metadata": chunk.metadata
    })

with open("chunks_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata_only, f, ensure_ascii=False, indent=2)

# 7. Выводим статистику и примеры
print("\n" + "="*60)
print(f"📊 СТАТИСТИКА:")
print(f"📄 Загружено документов: {len(documents)}")
print(f"✂️  Создано чанков: {len(chunks)}")
print(f"🔢 Размерность эмбеддингов: {embeddings.shape[1]}")
print(f"💾 Размер эмбеддингов в памяти: {embeddings.nbytes / 1024 / 1024:.2f} MB")
print("="*60)

# Показываем примеры
print("\n📝 ПРИМЕРЫ ЧАНКОВ С ЭМБЕДДИНГАМИ:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Чанк {i+1} ---")
    print(f"Источник: {chunk.metadata['source']}")
    print(f"Позиция (символ): {chunk.metadata.get('start_index', 'N/A')}")
    print(f"Текст: {chunk.page_content[:150]}...")
    print(f"Эмбеддинг (первые 5 значений): {embeddings[i][:5]}...")
    print(f"Норма эмбеддинга: {np.linalg.norm(embeddings[i]):.4f}")

print("\n✅ Готово! Файлы сохранены:")
print("   - chunks_with_embeddings.json (все данные + эмбеддинги)")
print("   - embeddings.npy (только эмбеддинги в numpy формате)")
print("   - chunks_metadata.json (тексты и метаданные без эмбеддингов)")