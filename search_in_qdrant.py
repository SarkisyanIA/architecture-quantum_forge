import json
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class QdrantSearcher:
    def __init__(self, host="localhost", port=6333, collection_name="knowledge_base_chunks"):
        """Инициализация поисковой системы"""
        print("🔌 Подключение к Qdrant...")
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        
        # Загружаем модель для эмбеддингов
        print("🔄 Загрузка модели all-MiniLM-L6-v2...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Готово!")
    
    def search(self, query: str, limit: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Поиск похожих чанков по текстовому запросу
        
        Args:
            query: Поисковый запрос
            limit: Количество результатов
            score_threshold: Порог релевантности (0-1)
        
        Returns:
            Список найденных чанков с оценками
        """
        # Генерируем эмбеддинг запроса
        query_embedding = self.model.encode(query)
        
        # Выполняем поиск в Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold,
        )
        
        # Форматируем результаты
        results = []
        for result in search_result:
            results.append({
                "score": result.score,
                "text": result.payload["text"],
                "source": result.payload["source"],
                "chunk_id": result.payload["chunk_id"],
                "metadata": {
                    "source": result.payload["source"],
                    "source_path": result.payload.get("source_path", ""),
                    "start_index": result.payload.get("start_index", -1)
                }
            })
        
        return results
    
    def batch_search(self, queries: List[str], limit: int = 3) -> Dict[str, List[Dict]]:
        """
        Пакетный поиск для нескольких запросов
        """
        results = {}
        for query in queries:
            print(f"🔍 Поиск: '{query}'")
            results[query] = self.search(query, limit=limit)
        return results
    
    def search_with_filter(self, query: str, source_filter: str = None, limit: int = 5):
        """
        Поиск с фильтрацией по источнику
        """
        from qdrant_client.models import Filter, FieldCondition, MatchText
        
        query_embedding = self.model.encode(query)
        
        # Создаем фильтр, если указан источник
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchText(text=source_filter)
                    )
                ]
            )
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            query_filter=query_filter
        )
        
        return [{"score": r.score, "text": r.payload["text"], "source": r.payload["source"]} 
                for r in search_result]

# Примеры использования
if __name__ == "__main__":
    # Создаем поисковую систему
    searcher = QdrantSearcher()
    
    # Интерактивный поиск
    print("\n" + "="*60)
    print("🔍 ИНТЕРАКТИВНЫЙ ПОИСК")
    print("Введите 'exit' для выхода")
    print("="*60)
    
    while True:
        query = input("\n📝 Ваш запрос: ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 До свидания!")
            break
        
        if not query:
            continue
        
        # Выполняем поиск
        results = searcher.search(query, limit=5, score_threshold=0.3)
        
        print(f"\n📊 Найдено {len(results)} результатов:")
        for i, result in enumerate(results, 1):
            print(f"\n--- Результат {i} (score: {result['score']:.4f}) ---")
            print(f"Источник: {result['source']}")
            print(f"Текст: {result['text'][:300]}...")
            if len(result['text']) > 300:
                print("...")
    
    # Пример поиска с фильтрацией
    print("\n" + "="*60)
    print("🔍 ПРИМЕРЫ ПОИСКА")
    print("="*60)
    
    # Пример 1: Обычный поиск
    query = "Что здесь написано?"  # Замените на свой запрос
    results = searcher.search(query, limit=3)
    print(f"\nЗапрос: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f} - {result['text'][:100]}...")
    
    # Пример 2: Пакетный поиск
    queries = ["ключевая идея", "пример использования"]
    batch_results = searcher.batch_search(queries, limit=2)
    for q, res in batch_results.items():
        print(f"\nЗапрос: '{q}'")
        for r in res:
            print(f"  - Score: {r['score']:.4f} | {r['text'][:80]}...")