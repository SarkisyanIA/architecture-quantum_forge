import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import uuid

# Конфигурация
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "knowledge_base_chunks"
EMBEDDING_DIM = 384  # Размерность all-MiniLM-L6-v2

# 1. Подключаемся к Qdrant
print("🔌 Подключение к Qdrant...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# 2. Проверяем подключение
try:
    client.get_collections()
    print("✅ Подключение успешно установлено")
except Exception as e:
    print(f"❌ Ошибка подключения: {e}")
    exit(1)

# 3. Удаляем коллекцию, если она существует (для пересоздания)
if client.collection_exists(COLLECTION_NAME):
    print(f"🗑️  Удаляем существующую коллекцию {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

# 4. Создаем новую коллекцию
print(f"📁 Создаем коллекцию {COLLECTION_NAME}...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_DIM,
        distance=Distance.COSINE,  # Используем косинусное расстояние
    ),
)

# 5. Загружаем данные из JSON
print("📖 Загрузка данных из chunks_with_embeddings.json...")
with open("chunks_with_embeddings.json", "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

# 6. Подготавливаем точки для вставки
print("🔄 Подготовка точек для вставки...")
points = []
for chunk in tqdm(chunks_data, desc="Обработка чанков"):
    # Создаем уникальный ID для точки
    point_id = chunk.get("chunk_id", str(uuid.uuid4()))
    
    # Извлекаем эмбеддинг
    embedding = np.array(chunk["embedding"], dtype=np.float32)
    
    # Подготавливаем метаданные (без эмбеддинга)
    payload = {
        "text": chunk["text"],
        "source": chunk["metadata"]["source"],
        "source_path": chunk["metadata"]["source_path"],
        "chunk_id": chunk["chunk_id"],
        "start_index": chunk["metadata"].get("start_index", -1),
    }
    
    points.append(
        PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload
        )
    )

# 7. Загружаем точки в Qdrant (батчами по 100 точек)
print("💾 Загрузка точек в Qdrant...")
batch_size = 100
for i in tqdm(range(0, len(points), batch_size), desc="Загрузка батчей"):
    batch = points[i:i + batch_size]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )

print(f"✅ Успешно загружено {len(points)} точек в коллекцию {COLLECTION_NAME}")

# 8. Проверяем статистику коллекции
collection_info = client.get_collection(collection_name=COLLECTION_NAME)
print(f"\n📊 Статистика коллекции:")
print(f"   - Имя: {COLLECTION_NAME}")
print(f"   - Количество векторов: {collection_info.vectors_count}")
print(f"   - Размерность: {collection_info.config.params.vectors.size}")
print(f"   - Метрика: {collection_info.config.params.vectors.distance}")