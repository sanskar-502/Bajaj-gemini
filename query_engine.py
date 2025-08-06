# query_engine.py

import re
from typing import List, Dict, Any, Optional, Tuple
from config import Config
from models import (
    QueryResponse, ClauseInfo, LogicTree, LogicTreeType, LogicCondition,
    SearchResult, QueryRequest
)
from vector_store import VectorStore
from llm_providers import get_llm_provider, LLMProvider

class QueryEngine:
    def __init__(self, vector_store: VectorStore):
        self.config = Config()
        self.vector_store = vector_store
        self.llm_provider: LLMProvider = get_llm_provider()
        model_name = getattr(self.llm_provider, 'model_name', 'N/A')
        print(f"[QueryEngine] Initialized with LLM Provider: {type(self.llm_provider).__name__} (Model: {model_name})")

    def process_query(self, request: QueryRequest) -> QueryResponse:
        search_results = self.vector_store.search(
            query=request.question,
            top_k=request.max_results,
            document_ids=request.document_ids
        )
        filtered_results = [r for r in search_results if r.score >= self.config.SIMILARITY_THRESHOLD]
        if not filtered_results:
            return self._create_no_results_response()
        
        clauses_used = self._create_clauses_from_search(filtered_results)
        answer, confidence = self._generate_final_answer(request.question, clauses_used)
        
        logic_tree = None
        if request.include_logic:
            logic_tree = self._generate_logic_tree_with_llm(request.question, clauses_used)

        intent, entities = self._extract_intent_and_entities(request.question)

        return QueryResponse(
            answer=answer,
            clauses_used=clauses_used,
            logic_tree=logic_tree,
            confidence=confidence,
            query_intent=intent,
            entities=entities
        )

    def _generate_final_answer(self, question: str, clauses: List[ClauseInfo]) -> Tuple[str, float]:
        context = "\n\n".join(
            f"Source Document ID: {c.document_id}\nClause Content:\n{c.text}" for c in clauses
        )
        prompt = f"Based only on the context below, answer the user's question.\n\nContext:\n---\n{context}\n---\n\nUser Question: {question}\n\nAnswer concisely. After your answer, on a new line, provide a confidence score like this: Confidence: [0.0-1.0]"
        
        response_text = self.llm_provider.generate_response(prompt)
        
        confidence_match = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", response_text, re.IGNORECASE)
        if confidence_match:
            confidence = float(confidence_match.group(1))
            answer = response_text.split(confidence_match.group(0))[0].strip()
        else:
            confidence = 0.5
            answer = response_text.strip()
        return answer, confidence

    def _generate_logic_tree_with_llm(self, question: str, clauses: List[ClauseInfo]) -> Optional[LogicTree]:
        context = "\n\n".join(f"Clause ID: {c.clause_id}\nClause Text: {c.text}" for c in clauses)
        prompt = f"Analyze the question and clauses to create a logic tree. Question: \"{question}\"\n\nClauses:\n---\n{context}\n---\n\nTask: Identify the logical conditions required. For each, determine if it is met and cite the source clause ID. Output a single JSON object matching this Pydantic format: {{\"type\": \"AND | OR\", \"conditions\": [{{\"condition\": \"...\", \"is_met\": boolean, \"source_clause_id\": \"...\"}}]}}"
        try:
            structured_response = self.llm_provider.generate_structured_response(prompt)
            return LogicTree.model_validate(structured_response)
        except Exception as e:
            print(f"Failed to generate or parse logic tree: {e}")
            return None

    def _create_clauses_from_search(self, search_results: List[SearchResult]) -> List[ClauseInfo]:
        return [
            ClauseInfo(
                title=res.metadata.get("title", "Untitled Section"),
                text=res.content,
                document_id=res.metadata.get("document_id", "unknown"),
                page=res.metadata.get("page"),
                relevance_score=res.score,
                clause_id=res.metadata.get("id"),
            ) for res in search_results
        ]

    def _extract_intent_and_entities(self, question: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        # Simple heuristic fallback
        return "General Inquiry", {}

    def _create_no_results_response(self) -> QueryResponse:
        return QueryResponse(
            answer="I could not find any relevant information in the documents to answer your question.",
            clauses_used=[],
            confidence=0.0,
        )

    def validate_query(self, question: str) -> Tuple[bool, str]:
        if not (10 < len(question.strip()) < 500):
            return False, "Query must be between 10 and 500 characters."
        return True, "Query is valid."