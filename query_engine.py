import json
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config import Config
from models import (
    QueryResponse, ClauseInfo, LogicTree, LogicTreeType, 
    SearchResult, QueryRequest
)
from vector_store import VectorStore
from llm_providers import LLMFactory, LLMProvider

class QueryEngine:
    """Handles query processing, LLM interactions, and response generation"""
    
    def __init__(self, vector_store: VectorStore, llm_provider: str = None):
        self.config = Config()
        self.vector_store = vector_store
        self.llm_provider = None
        
        # Initialize LLM provider
        self._initialize_llm_provider(llm_provider)
    
    def _initialize_llm_provider(self, provider_type: str = None):
        """Initialize the LLM provider"""
        try:
            self.llm_provider = LLMFactory.create_provider(provider_type)
            print(f"Initialized {self.llm_provider.get_model_info()['provider']} provider with model {self.llm_provider.get_model_info()['model']}")
        except Exception as e:
            available_providers = LLMFactory.get_available_providers()
            raise ValueError(f"Failed to initialize LLM provider. Available providers: {available_providers}. Error: {str(e)}")
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a user query and return a structured response
        
        Args:
            request: Query request object
            
        Returns:
            Structured query response
        """
        # Step 1: Extract query intent and entities
        intent, entities = self._extract_intent_and_entities(request.question)
        
        # Step 2: Retrieve relevant documents
        search_results = self.vector_store.search(
            query=request.question,
            top_k=request.max_results or self.config.TOP_K_RESULTS,
            document_ids=request.document_ids
        )
        
        # Step 3: Filter results by similarity threshold
        filtered_results = [
            result for result in search_results 
            if result.score >= self.config.SIMILARITY_THRESHOLD
        ]
        
        if not filtered_results:
            return self._create_no_results_response(request.question)
        
        # Step 4: Generate answer using LLM
        answer, confidence = self._generate_answer(request.question, filtered_results)
        
        # Step 5: Extract relevant clauses
        clauses_used = self._extract_clauses(filtered_results)
        
        # Step 6: Build logic tree
        logic_tree = self._build_logic_tree(request.question, filtered_results, clauses_used)
        
        return QueryResponse(
            answer=answer,
            clauses_used=clauses_used,
            logic_tree=logic_tree,
            confidence=confidence,
            query_intent=intent,
            entities=entities
        )
    
    def _extract_intent_and_entities(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """Extract query intent and entities"""
        prompt = f"""
        Analyze the following question and extract:
        1. The main intent (what the user is asking for)
        2. Key entities (people, organizations, conditions, etc.)
        
        Question: {question}
        
        Return your response as JSON:
        {{
            "intent": "description of what the user is asking",
            "entities": {{
                "actors": ["list of people/organizations"],
                "actions": ["list of actions/verbs"],
                "conditions": ["list of conditions/requirements"],
                "timeframes": ["list of time-related terms"],
                "documents": ["list of document types mentioned"]
            }}
        }}
        """
        
        try:
            result = self.llm_provider.generate_structured_response(prompt)
            
            if "error" in result:
                # Fallback to simple extraction
                return self._simple_intent_extraction(question), {}
            
            return result.get("intent", "unknown"), result.get("entities", {})
        except Exception as e:
            # Fallback to simple extraction
            return self._simple_intent_extraction(question), {}
    
    def _simple_intent_extraction(self, question: str) -> str:
        """Simple intent extraction using keyword matching"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["can", "allowed", "permitted"]):
            return "permission_check"
        elif any(word in question_lower for word in ["what happens", "consequence", "result"]):
            return "consequence_analysis"
        elif any(word in question_lower for word in ["show", "find", "list", "sections"]):
            return "information_retrieval"
        elif any(word in question_lower for word in ["covered", "include", "exclude"]):
            return "coverage_check"
        elif any(word in question_lower for word in ["terminate", "fire", "dismiss"]):
            return "termination_analysis"
        else:
            return "general_inquiry"
    
    def _generate_answer(self, question: str, search_results: List[SearchResult]) -> Tuple[str, float]:
        """Generate answer using LLM"""
        # Prepare context from search results
        context = self._prepare_context(search_results)
        
        prompt = f"""
        You are an intelligent legal-insurance-HR-compliance document analyst assistant.
        
        Based on the following context from legal documents, answer the user's question clearly and factually.
        
        User Question: {question}
        
        Context from documents:
        {context}
        
        Instructions:
        1. Answer only based on the provided context
        2. If the context doesn't contain enough information, say "The documents do not provide a clear answer to your question."
        3. Be specific and cite relevant sections when possible
        4. If there are conditions or exceptions, clearly state them
        5. Provide a confidence score (0-1) based on how well the context answers the question
        
        Answer:
        """
        
        try:
            result = self.llm_provider.generate_response(prompt)
            
            # Extract confidence score if present
            confidence = self._extract_confidence(result)
            
            # Clean up the answer
            answer = self._clean_answer(result)
            
            return answer, confidence
        except Exception as e:
            return "I encountered an error while processing your question. Please try again.", 0.0
    
    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """Prepare context from search results"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Document {i} (Score: {result.score:.2f}):")
            context_parts.append(f"Source: {result.source_document}")
            context_parts.append(f"Content: {result.content}")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _extract_confidence(self, llm_response: str) -> float:
        """Extract confidence score from LLM response"""
        # Look for confidence score in the response
        confidence_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', llm_response.lower())
        if confidence_match:
            try:
                return float(confidence_match.group(1))
            except ValueError:
                pass
        
        # Default confidence based on response quality
        if "do not provide a clear answer" in llm_response.lower():
            return 0.1
        elif len(llm_response) > 100:
            return 0.8
        else:
            return 0.5
    
    def _clean_answer(self, llm_response: str) -> str:
        """Clean up the LLM response"""
        # Remove confidence score from the answer
        answer = re.sub(r'confidence[:\s]*[0-9]*\.?[0-9]+', '', llm_response, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer
    
    def _extract_clauses(self, search_results: List[SearchResult]) -> List[ClauseInfo]:
        """Extract relevant clauses from search results"""
        clauses = []
        
        for result in search_results:
            # Extract clause title from content
            title = self._extract_clause_title(result.content)
            
            clause = ClauseInfo(
                title=title,
                text=result.content,
                document=result.source_document,
                page=result.metadata.get("page", None),
                relevance_score=result.score,
                clause_id=result.metadata.get("chunk_id", None),
                section=result.metadata.get("section", None)
            )
            clauses.append(clause)
        
        return clauses
    
    def _extract_clause_title(self, content: str) -> str:
        """Extract clause title from content"""
        # Look for common clause title patterns
        patterns = [
            r'^([A-Z][A-Za-z\s]+):',  # Title followed by colon
            r'^([A-Z][A-Za-z\s]+)\s*$',  # Title on its own line
            r'Section\s+(\d+[\.\d]*)\s*[-:]\s*([A-Za-z\s]+)',  # Section number and title
            r'Clause\s+(\d+[\.\d]*)\s*[-:]\s*([A-Za-z\s]+)',  # Clause number and title
        ]
        
        lines = content.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    if len(match.groups()) == 2:
                        return f"{match.group(1)} - {match.group(2)}"
                    else:
                        return match.group(1).strip()
        
        # Fallback: use first sentence as title
        first_sentence = content.split('.')[0]
        if len(first_sentence) > 50:
            first_sentence = first_sentence[:50] + "..."
        return first_sentence
    
    def _build_logic_tree(self, question: str, search_results: List[SearchResult], 
                         clauses: List[ClauseInfo]) -> LogicTree:
        """Build logic tree for reasoning"""
        # Analyze conditions in the question
        conditions = self._extract_conditions(question, clauses)
        
        if len(conditions) == 1:
            logic_tree = LogicTree(
                type=LogicTreeType.CONDITIONAL,
                conditions=conditions,
                result=True
            )
        elif len(conditions) > 1:
            # Determine if conditions are AND or OR based on question
            question_lower = question.lower()
            if any(word in question_lower for word in ["and", "both", "all"]):
                logic_type = LogicTreeType.AND
            elif any(word in question_lower for word in ["or", "either", "any"]):
                logic_type = LogicTreeType.OR
            else:
                logic_type = LogicTreeType.AND  # Default to AND
            
            logic_tree = LogicTree(
                type=logic_type,
                conditions=conditions,
                result=True
            )
        else:
            logic_tree = LogicTree(
                type=LogicTreeType.CONDITIONAL,
                conditions=["No specific conditions identified"],
                result=True
            )
        
        return logic_tree
    
    def _extract_conditions(self, question: str, clauses: List[ClauseInfo]) -> List[str]:
        """Extract conditions from question and clauses"""
        conditions = []
        
        # Extract conditions from question
        question_conditions = self._extract_question_conditions(question)
        conditions.extend(question_conditions)
        
        # Extract conditions from clauses
        for clause in clauses:
            clause_conditions = self._extract_clause_conditions(clause.text)
            conditions.extend(clause_conditions)
        
        return list(set(conditions))  # Remove duplicates
    
    def _extract_question_conditions(self, question: str) -> List[str]:
        """Extract conditions from the question"""
        conditions = []
        
        # Look for conditional phrases
        conditional_patterns = [
            r'if\s+([^,]+)',
            r'when\s+([^,]+)',
            r'provided\s+that\s+([^,]+)',
            r'as\s+long\s+as\s+([^,]+)',
            r'only\s+if\s+([^,]+)',
        ]
        
        for pattern in conditional_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            conditions.extend(matches)
        
        return conditions
    
    def _extract_clause_conditions(self, clause_text: str) -> List[str]:
        """Extract conditions from clause text"""
        conditions = []
        
        # Look for conditional statements in clause
        conditional_patterns = [
            r'if\s+([^,\.]+)',
            r'provided\s+that\s+([^,\.]+)',
            r'unless\s+([^,\.]+)',
            r'when\s+([^,\.]+)',
            r'where\s+([^,\.]+)',
        ]
        
        for pattern in conditional_patterns:
            matches = re.findall(pattern, clause_text, re.IGNORECASE)
            conditions.extend(matches)
        
        return conditions
    
    def _create_no_results_response(self, question: str) -> QueryResponse:
        """Create response when no relevant results are found"""
        return QueryResponse(
            answer="The documents do not provide a clear answer to your question. Please try rephrasing your query or upload additional relevant documents.",
            clauses_used=[],
            logic_tree=LogicTree(
                type=LogicTreeType.CONDITIONAL,
                conditions=["No relevant documents found"],
                result=False
            ),
            confidence=0.0,
            query_intent="no_results",
            entities={}
        )
    
    def get_similar_questions(self, question: str, top_k: int = 5) -> List[str]:
        """Get similar questions from the vector store"""
        # This could be implemented by storing previous questions and their embeddings
        # For now, return empty list
        return []
    
    def validate_query(self, question: str) -> Tuple[bool, str]:
        """Validate if a query is appropriate for the system"""
        if len(question.strip()) < 10:
            return False, "Query is too short. Please provide more details."
        
        if len(question) > 500:
            return False, "Query is too long. Please keep it under 500 characters."
        
        # Check for inappropriate content
        inappropriate_words = ["hack", "illegal", "fraud", "steal"]
        if any(word in question.lower() for word in inappropriate_words):
            return False, "Query contains inappropriate content."
        
        return True, "Query is valid." 