import argparse
import os
from simple_rag import LocalRAG

def main():
    parser = argparse.ArgumentParser(description="RAG система с Qdrant и Ollama")
    parser.add_argument("--question", "-q", type=str, help="Вопрос к системе")
    parser.add_argument("--interactive", "-i", action="store_true", help="Интерактивный режим")
    
    args = parser.parse_args()
    
    # Инициализация RAG
    rag = LocalRAG()
    
    if args.interactive:
        rag.interactive_mode()
    elif args.question:
        result = rag.ask(args.question)
        print(f"\n🤖 Ответ:\n{result['answer']}")
        if result['sources']:
            print(f"\n📚 Источники:")
            for src in result['sources']:
                print(f"  - {src['source']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()