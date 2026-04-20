import sys
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class QdrantQA:
    def __init__(self, host="localhost", port=6333, collection_name="knowledge_base_chunks"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def ask(self, question: str, top_k: int = 3) -> list:
        """Задает вопрос и возвращает наиболее релевантные ответы"""
        # Генерируем эмбеддинг вопроса
        question_embedding = self.model.encode(question)
        
        # Используем query_points вместо search (для qdrant-client 1.x)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=question_embedding.tolist(),
            limit=top_k,
            score_threshold=0.3
        )
        
        return response.points
    
    def pretty_print(self, question: str, results):
        """Красиво выводит результаты"""
        print("\n" + "="*80)
        print(f"❓ Вопрос: {question}")
        print("="*80)
        
        if not results:
            print("❌ Не найдено релевантных ответов.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n📌 Ответ {i} (релевантность: {result.score:.4f}):")
            print(f"   Источник: {result.payload.get('source', 'Неизвестен')}")
            print(f"   Текст: {result.payload['text'][:500]}")
            if len(result.payload['text']) > 500:
                print("   ...")
            print("-"*80)

# Основная логика
if __name__ == "__main__":
    qa = QdrantQA()
    
    # Список вопросов
    questions = [
        "Любит битвы и остался непобедимым", #1
        "Носит маску и прячет лицо", #16
        "Представитель элементалев" #7
    ]
    
    # Задаем вопросы
    for question in questions:
        try:
            results = qa.ask(question, top_k=3)
            qa.pretty_print(question, results)
        except Exception as e:
            print(f"❌ Ошибка при поиске '{question}': {e}")