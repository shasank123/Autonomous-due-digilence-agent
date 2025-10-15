from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Optional, Dict, Any
import logging

class ProductionRagsystem:

    #Production RAG system with advanced features
    def __init__(self,
                persist_directory: str = "./data/vector_stores/financial_data",
                embedding_type: str = "huggingface",
                chunk_size : int = 1000,
                chunk_overlap : int = 200
                ):
        self.persist_directory = persist_directory
        self.embedding_type = embedding_type
        self.logger = logging.getLogger(__name__)

        # Initialize embeddings
        if embedding_type == "openai":
            self.embeddings = OpenAIEmbeddings(
                model_name="text-embedding-3-small",
                open_api_key = os.getenv("OPEN_API_KEY")
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name= "sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
            separators= ["\n", "\n\n", ""," ","."]
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory= persist_directory,
            embedding_function= self.embeddings
        )

        print(f" ✅ Production RAG System initialized at {persist_directory}")

    
    def add_company_data(self, documents: List[Document]) -> List[str]:
        
        #Add documents with intelligent chunking and error handling
        if not documents:
            self.logger.warning("No documents provided to add_company_data")
            return []
        
        try:
            # Split documents into chunks for better retrieval
            all_chunks = []
            for doc in documents:
                chunks = self.text_splitter.split_documents(doc)
                all_chunks.extend(chunks)
            self.logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents(all_chunks)
            self.vector_store.persist()

            self.logger.info(f"✅ Added {len(doc_ids)} document chunks to vector store")
            return doc_ids
        
        except Exception as e:
            self.logger.error(f"❌ Failed to add documents: {e}")
            return []
        

    def query(self, 
              question: str,
              k : int = 5,
              company: Optional[str] = None,
              metric_type : Optional[str] = None,
              score_threshold: float = 0.7) -> List[Document]:
        
        #Advanced querying with filters and score thresholding
        
        search_kwargs = {"k": k}
        filters = {}

        # Build filters
        if company:
            filters["company"] = company.upper()

        if metric_type:
            filters["doc_type"] = metric_type

        if filters:
            search_kwargs["filters"] = filters
             
        self.logger.info(f"🔍 Query: '{question}' | Filters: {filters}")

        try:
            # Get results with scores
            scores_with_results = self.query_with_similarity_scores(
            question,
            **search_kwargs
        )
            # Filter by score threshold
            filtered_results = [
                doc for doc, score in scores_with_results
                if score >= score_threshold
            ]

            self.logger.info(f""✅ Found {len(filtered_results)} relevant documents (score >= {score_threshold})"")

            return filtered_results
        
        except Exception as e:
            self.logger.error(f"❌ Query failed: {e}")
            return []
        

    def get_company_metrics(self, company: str) -> List[str]:

        #Get all available metrics for a company

        try:
            # Query for company overview to see available metrics
            results = self.query(
                f"metrics financial data {company}",
                company = company,
                k = 20
            )

            metrics = set()
            for doc in results:
                if "metric" in doc.metadata:
                    metrics.add(doc.metadata["metric"])

            return sorted(list(metrics))
        
        except Exception as e:
            self.logger.error(f"❌ Failed to get company metrics: {e}")
            return []


    def clear_company_data(self, company: str) -> bool:

        #Remove all data for a specific company
        try:
            self.vector_store._collection.delete(
                where= {"company": company.upper()}
            )

            self.vector_store.persist()
            self.logger.info(f"✅ Cleared all data for {company}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to clear company data: {e}")
            return False
        
    def get_stats(self) -> Dict[str, Any]:

        #Get system statistics
        try:
            count = self.vector_store._collection.count()
            return {
                "total documents": count,
                "embedding_type": self.embedding_type,
                "persist_directory": self.persist_directory
            }
        
        except Exception as e:
            self.logger.error(f"❌ Failed to get stats: {e}")
            return {}

    # Production test function        
    def test_production_rag():
        from data.collectors.sec_edgar import SECDataCollector
        from data.processors.document_parser import DocumentProcessor

        print("🧪 Testing Production RAG System...")
        
        # Initialize components
        collector = SECDataCollector()
        processor = DocumentProcessor()
        rag_system = ProductionRagsystem()

        test_companies = ["AAPL", "MSFT", "TSLA"]

        for ticker in test_companies[:1]:  # testing with one company
            print(f"\n🔧 Testing with {ticker}...")

            # Fetch data
            print("1. Fetching SEC data...")
            company_data = collector.company_facts(ticker)

            if not company_data:
                print(f"❌ Failed to fetch data for {ticker}")
                continue

            #Process into documents
            print("2. Processing documents...")
            documents = processor.process_sec_facts(company_data, ticker)

            if not documents:
                print(f"❌ No documents created for {ticker}")
                continue

            #Add to RAG system
            print("3. Adding to RAG system...")
            doc_ids = rag_system.add_documents(documents)

            # Test queries
            print("4. Testing queries...")

            # Financial query
            financial_results = rag_system.query(
                f" what is {ticker}'s revenue and profit",
                company = ticker
            )
            print(f"financial results: {len(financial_results)} documents")

            # Ratio query
            ratio_results = rag_system.query(
                f"financial ratios performance",
                company = ticker,
                metric_type = "financial_ratio"
            )
            print(f"ratio results: {len(ratio_results)} doucments")

            #Get metrics
            metrics = rag_system.get_company_metrics(ticker)
            print(f" Available metrics: {len(metrics)}")

            #Get stats
            stats = rag_system.get_stats()
            print(f" System stats: {stats}")

    print("\n✅ Production RAG test completed!")


    if __name__ == "__main":
        test_production_rag()
            
    






        

        



        
  
    




    
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
            )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        return self.vector_store.add_documents(documents)
    
    def query(self,
              question: str,
              k: int = 5,
              filter_metadata: Optional[dict] = None
              ) -> List[Document]:
        search_kwargs = {"k": k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata
            return self.vector_store.similarity_search(question, **search_kwargs)
        
    def query_with_similarity_scores(
            self,
            question: str,
            k: int = 5) -> List[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(question, k=k)

    def document_count(self) -> int:
        return self.vector_store._collection.count()

    def persist(self):
        self.vector_store.persist()


    
              
              

        
       