"""
Simple RAG Pipeline for Local Testing
- In-memory Qdrant
- OpenRouter.ai support
- All advanced RAG features
- No frontend needed
- CLI interface

Installation:
pip install qdrant-client sentence-transformers openai pypdf2 python-dotenv

Usage:
python simple_rag.py
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import hashlib
from dotenv import load_dotenv

# Core libraries
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
import PyPDF2
import openai

# Load environment variables
load_dotenv()

@dataclass
class Document:
    """Document with metadata"""
    text: str
    metadata: Dict
    chunk_id: int = 0

@dataclass
class RetrievalResult:
    """Result from retrieval"""
    text: str
    score: float
    metadata: Dict

class DocumentProcessor:
    """Process PDFs and create chunks"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            print(f"✓ Loaded PDF: {pdf_path} ({len(text)} characters)")
            return text
        except Exception as e:
            print(f"✗ Error loading PDF: {e}")
            return ""
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Document]:
        """Create semantic chunks with overlap"""
        # Split by sentences (simple approach)
        sentences = text.replace('\n', ' ').split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            sentence_size = len(words)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                # Add context from previous chunk
                context = ""
                if chunks:
                    prev_text = chunks[-1].text
                    context = f"Previous context: {prev_text[:100]}...\n\n"
                
                chunks.append(Document(
                    text=context + chunk_text,
                    metadata={
                        **metadata,
                        'chunk_id': len(chunks),
                        'position': i / len(sentences)
                    },
                    chunk_id=len(chunks)
                ))
                
                # Keep overlap
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_sentences
                current_size = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Document(
                text=chunk_text,
                metadata={
                    **metadata,
                    'chunk_id': len(chunks),
                    'position': 1.0
                },
                chunk_id=len(chunks)
            ))
        
        print(f"✓ Created {len(chunks)} chunks")
        return chunks

class EmbeddingService:
    """Generate embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded (dimension: {self.dimension})")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

class VectorStore:
    """In-memory Qdrant vector store"""
    
    def __init__(self, collection_name: str = "cbse_docs", dimension: int = 384):
        print("Initializing in-memory Qdrant...")
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        
        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
        print(f"✓ Created collection: {collection_name}")
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents to vector store"""
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id = hashlib.md5(f"{doc.text[:50]}_{i}".encode()).hexdigest()
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'text': doc.text,
                    'chunk_id': doc.chunk_id,
                    **doc.metadata
                }
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"✓ Added {len(points)} documents to vector store")
    
    def search(self, query_embedding: List[float], filters: Dict = None, top_k: int = 5) -> List[RetrievalResult]:
        """Search vector store"""
        # Build filter
        filter_obj = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                filter_obj = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=filter_obj,
            limit=top_k
        )
        
        return [
            RetrievalResult(
                text=r.payload['text'],
                score=r.score,
                metadata={k: v for k, v in r.payload.items() if k != 'text'}
            )
            for r in results
        ]

class Reranker:
    """Cross-encoder reranking"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("✓ Reranker loaded")
    
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 3) -> List[RetrievalResult]:
        """Rerank results using cross-encoder"""
        if not results:
            return []
        
        # Create pairs
        pairs = [[query, r.text] for r in results]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Update scores and sort
        for result, score in zip(results, scores):
            result.score = float(score)
        
        reranked = sorted(results, key=lambda x: x.score, reverse=True)
        print(f"✓ Reranked to top {top_k} results")
        return reranked[:top_k]

class OpenRouterLLM:
    """OpenRouter.ai LLM integration"""
    
    def __init__(self, api_key: str, model: str = "deepseek/deepseek-chat-v3.1:free"):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        print(f"✓ Connected to OpenRouter.ai (model: {model})")
    
    def generate(self, query: str, context: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """Generate response using LLM"""
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_message = f"""Context from CBSE curriculum:

{context}

Question: {query}

Please provide a detailed answer based on the context above. Include relevant examples and cite sources when possible."""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating response: {e}"

class HyDEQueryExpander:
    """Hypothetical Document Embeddings for query expansion"""
    
    def __init__(self, llm: OpenRouterLLM):
        self.llm = llm
    
    def expand_query(self, query: str, grade: int, subject: str) -> str:
        """Generate hypothetical document for better retrieval"""
        prompt = f"""For a Grade {grade} {subject} student, write a brief educational passage that would answer this question:

Question: {query}

Write 2-3 sentences as if from a textbook:"""
        
        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            
            hypothetical_doc = response.choices[0].message.content
            print(f"✓ Generated HyDE document")
            return hypothetical_doc
        except:
            return query

class RAGPipeline:
    """Complete RAG pipeline with all features"""
    
    def __init__(
        self,
        openrouter_api_key: str,
        llm_model: str = "deepseek/deepseek-chat-v3.1:free",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_reranking: bool = True,
        use_hyde: bool = True
    ):
        print("\n" + "="*60)
        print("Initializing RAG Pipeline")
        print("="*60 + "\n")
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService(embedding_model)
        self.vector_store = VectorStore(dimension=self.embedding_service.dimension)
        self.llm = OpenRouterLLM(openrouter_api_key, llm_model)
        
        self.use_reranking = use_reranking
        self.use_hyde = use_hyde
        
        if use_reranking:
            self.reranker = Reranker()
        
        if use_hyde:
            self.hyde_expander = HyDEQueryExpander(self.llm)
        
        print("\n" + "="*60)
        print("✓ RAG Pipeline Ready!")
        print("="*60 + "\n")
    
    def add_document(self, pdf_path: str, metadata: Dict):
        """Add a document to the knowledge base"""
        print(f"\nProcessing document: {pdf_path}")
        print("-" * 60)
        
        # Extract text
        text = self.doc_processor.load_pdf(pdf_path)
        if not text:
            print("✗ Failed to extract text")
            return
        
        # Create chunks
        chunks = self.doc_processor.chunk_text(text, metadata)
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
        
        print(f"✓ Document added successfully\n")
    
    def query(
        self,
        question: str,
        grade: int,
        subject: str,
        top_k: int = 5,
        rerank_k: int = 3
    ) -> Dict:
        """Query the RAG system"""
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"Grade: {grade} | Subject: {subject}")
        print('='*60 + "\n")
        
        # Step 1: Query expansion with HyDE
        search_query = question
        if self.use_hyde:
            print("Step 1: Query Expansion (HyDE)")
            print("-" * 60)
            hypothetical_doc = self.hyde_expander.expand_query(question, grade, subject)
            search_query = hypothetical_doc
            print(f"HyDE document: {hypothetical_doc[:100]}...\n")
        
        # Step 2: Embed query
        print("Step 2: Embedding Query")
        print("-" * 60)
        query_embedding = self.embedding_service.embed_single(search_query)
        print("✓ Query embedded\n")
        
        # Step 3: Vector search
        print(f"Step 3: Vector Search (top {top_k})")
        print("-" * 60)
        results = self.vector_store.search(
            query_embedding,
            filters={'grade': grade, 'subject': subject},
            top_k=top_k
        )
        print(f"✓ Found {len(results)} results\n")
        
        # Step 4: Reranking
        if self.use_reranking and results:
            print(f"Step 4: Reranking (top {rerank_k})")
            print("-" * 60)
            results = self.reranker.rerank(question, results, top_k=rerank_k)
            print(f"✓ Reranked to {len(results)} results\n")
        
        # Step 5: Prepare context
        print("Step 5: Preparing Context")
        print("-" * 60)
        context = "\n\n".join([
            f"[Source {i+1}] (Relevance: {r.score:.3f})\n{r.text}"
            for i, r in enumerate(results)
        ])
        print(f"✓ Context prepared ({len(context)} characters)\n")
        
        # Step 6: Generate response
        print("Step 6: Generating Response")
        print("-" * 60)
        system_prompt = f"""You are an expert CBSE {subject} tutor for Grade {grade} students.
Provide clear, accurate answers based on the curriculum.
Use examples appropriate for Grade {grade} level.
Always cite your sources using [Source X] notation."""
        
        response = self.llm.generate(
            query=question,
            context=context,
            system_prompt=system_prompt
        )
        print("✓ Response generated\n")
        
        # Return results
        return {
            'question': question,
            'answer': response,
            'sources': [
                {
                    'text': r.text[:200] + '...',
                    'score': r.score,
                    'metadata': r.metadata
                }
                for r in results
            ],
            'metadata': {
                'grade': grade,
                'subject': subject,
                'num_sources': len(results),
                'used_hyde': self.use_hyde,
                'used_reranking': self.use_reranking
            }
        }

def print_result(result: Dict):
    """Pretty print query result"""
    print("\n" + "="*60)
    print("RESULT")
    print("="*60 + "\n")
    
    print("ANSWER:")
    print("-" * 60)
    print(result['answer'])
    print()
    
    print("SOURCES:")
    print("-" * 60)
    for i, source in enumerate(result['sources'], 1):
        print(f"\n[{i}] Relevance: {source['score']:.3f}")
        print(f"    Chapter: {source['metadata'].get('chapter', 'N/A')}")
        print(f"    {source['text']}")
    
    print("\n" + "="*60 + "\n")

def main():
    """Main function - CLI interface"""
    
    # Get API key
    #api_key = os.getenv("OPENROUTER_API_KEY")
    api_key = "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"
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment")
        print("Please create a .env file with: OPENROUTER_API_KEY=your_key_here")
        print("Get your key from: https://openrouter.ai/keys")
        return
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        openrouter_api_key=api_key,
        llm_model= "deepseek/deepseek-chat-v3.1:free",#"openai/gpt-4-turbo",  # or "anthropic/claude-3-opus", "google/gemini-pro"
        use_reranking=True,
        use_hyde=True
    )
    
    # Example usage
    print("\n" + "="*60)
    print("RAG PIPELINE - EXAMPLE USAGE")
    print("="*60 + "\n")
    
    # Option 1: Add documents from PDF
    print("Option 1: Add PDF documents")
    print("-" * 60)
    print("rag.add_document('math_grade10.pdf', {")
    print("    'subject': 'Mathematics',")
    print("    'grade': 10,")
    print("    'chapter': 'Quadratic Equations'")
    print("})")
    print()
    
    # Option 2: Add sample text
    print("Option 2: Adding sample text for demo...")
    print("-" * 60)
    sample_text = """
    Quadratic Equations - Grade 10 Mathematics
    
    A quadratic equation is a polynomial equation of degree 2. The standard form is ax² + bx + c = 0, where a ≠ 0.
    
    The quadratic formula is used to find the roots: x = (-b ± √(b²-4ac)) / 2a
    
    The discriminant D = b² - 4ac determines the nature of roots:
    - If D > 0: Two distinct real roots
    - If D = 0: Two equal real roots
    - If D < 0: No real roots (complex roots)
    
    Example: Solve x² - 5x + 6 = 0
    Here, a=1, b=-5, c=6
    D = (-5)² - 4(1)(6) = 25 - 24 = 1 > 0
    So we have two distinct real roots.
    
    Using the quadratic formula:
    x = (5 ± √1) / 2 = (5 ± 1) / 2
    x = 3 or x = 2
    
    Verification: (x-2)(x-3) = x² - 5x + 6 = 0 ✓
    """
    
    # Create a temporary document
    chunks = rag.doc_processor.chunk_text(sample_text, {
        'subject': 'Mathematics',
        'grade': 10,
        'chapter': 'Quadratic Equations',
        'topic': 'Solving Quadratic Equations'
    })
    
    texts = [c.text for c in chunks]
    embeddings = rag.embedding_service.embed_texts(texts)
    rag.vector_store.add_documents(chunks, embeddings)
    
    print("✓ Sample content added\n")
    
    # Query the system
    print("Querying the system...")
    result = rag.query(
        question="Explain how to solve quadratic equations and give an example",
        grade=10,
        subject="Mathematics"
    )
    
    # Print result
    print_result(result)
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60 + "\n")
    print("Commands:")
    print("  query <question> - Ask a question")
    print("  add <pdf_path> - Add a PDF document")
    print("  quit - Exit")
    print()
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if cmd == "quit":
                break
            
            elif cmd.startswith("query "):
                question = cmd[6:]
                result = rag.query(question, grade=10, subject="Mathematics")
                print_result(result)
            
            elif cmd.startswith("add "):
                pdf_path = cmd[4:]
                rag.add_document(pdf_path, {
                    'subject': 'Mathematics',
                    'grade': 10,
                    'chapter': 'Sample Chapter'
                })
            
            else:
                print("Unknown command. Use: query <question>, add <pdf>, or quit")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":

    main()
