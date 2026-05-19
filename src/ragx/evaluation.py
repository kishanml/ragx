"""
Evaluation module for RAG pipeline
"""

import json
import numpy as np
from collections import defaultdict
from ranx import Qrels, Run, evaluate


# TIER 1: RETRIEVAL METRICS (using ranx)

def evaluate_tier1_retrieval(retrieved_ids,  relevant_ids, retrieved_chunks, k_values = [1, 3, 5]):
    """
    evaluate retrieval quality using ranx library.
    
    metrics:
    - Recall@K: Proportion of relevant docs retrieved
    - Precision@K: Proportion of retrieved docs that are relevant
    - MAP: Mean Average Precision
    - MRR: Mean Reciprocal Rank
    - nDCG@K: Normalized Discounted Cumulative Gain
    - Hit Rate: Binary - was any relevant doc retrieved?
    - Diversity@K: Topical diversity based on headings/pages
    
    args:
        retrieved_ids: List of retrieved chunk IDs (ranked)
        relevant_ids: List of ground truth chunk IDs
        retrieved_chunks: List of chunk dicts (for diversity calculation)
        k_values: List of K values for @K metrics
        
    Returns:
        Dict of metric_name -> score
    """
    # convert to ranx format
    # qrels: {query_id: {doc_id: relevance_score}}
    # run: {query_id: {doc_id: rank_score}}
    
    query_id = "q1"
    
    # create ground truth (qrels)
    qrels_dict = {query_id: {str(doc_id): 1 for doc_id in relevant_ids}}
    qrels = Qrels(qrels_dict)
    
    # create predictions (run) - assign decreasing scores by rank
    run_dict = {query_id: {str(retrieved_ids[i]): float(len(retrieved_ids) - i) for i in range(len(retrieved_ids))}}
    run = Run(run_dict)
    
    metrics = {}
    
    # evaluate using ranx
    for k in k_values:
        # recall@K
        recall = evaluate(qrels, run, f'recall@{k}')
        metrics[f'recall@{k}'] = recall
        
        # precision@K
        precision = evaluate(qrels, run, f'precision@{k}')
        metrics[f'precision@{k}'] = precision
        
        # nDCG@K (Normalized Discounted Cumulative Gain)
        ndcg = evaluate(qrels, run, f'ndcg@{k}')
        metrics[f'ndcg@{k}'] = ndcg
    
    # MAP (Mean Average Precision)
    map_score = evaluate(qrels, run, 'map')
    metrics['map'] = map_score
    
    # MRR (Mean Reciprocal Rank)
    mrr_score = evaluate(qrels, run, 'mrr')
    metrics['mrr'] = mrr_score
    
    # hit Rate (manual calculation)
    relevant_set = set(relevant_ids)
    hit_rate = 1.0 if any(doc_id in relevant_set for doc_id in retrieved_ids) else 0.0
    metrics['hit_rate'] = hit_rate
    
    # diversity@K (manual calculation)
    if retrieved_chunks:
        for k in k_values:
            diversity = calculate_diversity_at_k(retrieved_chunks, k)
            metrics[f'diversity@{k}'] = diversity
    
    return metrics


def calculate_diversity_at_k(retrieved_chunks, k):
    """
    calculated topical diversity using heading and page distribution.
    
    formula: (# unique headings + # unique pages) / (2 * k)
    range: 0.0 to 1.0
    
    rrgs:
        retrieved_chunks: List of chunk dicts with 'heading' and 'page' keys
        k: Number of top chunks to consider
        
    returns:
        Diversity score (0.0 to 1.0)
    """
    if not retrieved_chunks or k == 0:
        return 0.0
    
    chunks_k = retrieved_chunks[:k]
    
    unique_headings = len(set(chunk.get('heading', 'Unknown') for chunk in chunks_k))
    unique_pages = len(set(chunk.get('page', 0) for chunk in chunks_k))
    
    max_diversity = 2 * k
    actual_diversity = unique_headings + unique_pages
    
    return actual_diversity / max_diversity


# TIER 2: GENERATION QUALITY METRICS

def parse_llm_json_response(response):
    """
    parse JSON from LLM response, handling markdown code blocks
    
    args:
        response: Raw LLM response string
        
    returns:
        Parsed JSON dict
    """
    response = response.strip()
    
    # removed markdown code blocks
    if response.startswith('```json'):
        response = response.split('```json')[1].split('```')[0].strip()
    elif response.startswith('```'):
        response = response.split('```')[1].split('```')[0].strip()
    
    return json.loads(response)


def evaluate_context_relevance(query, contexts, judge_llm):
    """
    context relevance (C|Q): Are retrieved contexts relevant to the query?
    
    rubric:
    - highly relevant (1.0): Context directly addresses the query
    - partially relevant (0.5): Context has some related information
    - not relevant (0.0): Context is unrelated to the query
    
    args:
        query: User query
        contexts: List of retrieved context strings
        judge_llm: LLM judge instance
        
    Returns:
        Dict with relevance score and details
    """
    relevance_scores = []
    
    for i, context in enumerate(contexts):
        prompt = f"""You are evaluating the relevance of a retrieved context to a user query.

            RUBRIC:
            - Score 1.0 (Highly Relevant): Context directly addresses the query and contains key information needed to answer it
            - Score 0.5 (Partially Relevant): Context contains some information related to the query but misses key aspects
            - Score 0.0 (Not Relevant): Context is completely unrelated to the query

            Query: {query}

            Context: {context}

            Evaluate the relevance of this context to the query.
            Respond with ONLY a JSON object:
            {{'relevant': true/false, 'score': 0.0/0.5/1.0, 'explanation': 'brief explanation'}}"""

        try:
            response = judge_llm.invoke(prompt).content
            result = parse_llm_json_response(response)
            score = result.get('score', 1.0 if result.get('relevant', False) else 0.0)
            relevance_scores.append(score)
        except Exception as e:
            print(f'Error evaluating context {i}: {e}')
            relevance_scores.append(0.0)
    
    return {
        'context_relevance_score': np.mean(relevance_scores) if relevance_scores else 0.0,
        'relevant_contexts_count': sum(1 for s in relevance_scores if s >= 0.5),
        'total_contexts': len(contexts)
    }


def evaluate_faithfulness(answer, contexts, judge_llm):
    """
    Faithfulness (A|C): Is the answer faithful to the provided contexts?
    
    rubric:
    - fully faithful (1.0): All claims in answer are supported by contexts
    - partially faithful (0.5): Most claims supported, minor unsupported details
    - not faithful (0.0): Answer contains significant hallucinations or unsupported claims
    
    args:
        answer: Generated answer
        contexts: List of context strings used to generate answer
        judge_llm: LLM judge instance
        
    returns:
        Dict with faithfulness score and details
    """
    context_combined = '\n\n'.join(contexts)
    
    prompt = f"""You are evaluating whether an answer is faithful to the provided contexts.

        RUBRIC:
        - Score 1.0 (Fully Faithful): Every claim in the answer is directly supported by the contexts. No hallucinations.
        - Score 0.5 (Partially Faithful): Most major claims are supported, but there are minor unsupported details or slight extrapolations.
        - Score 0.0 (Not Faithful): Answer contains significant claims not supported by contexts, or contradicts the contexts.

        Contexts:
        {context_combined}

        Answer:
        {answer}

        Evaluate the faithfulness of the answer to the contexts.
        Respond with ONLY a JSON object:
        {{'faithful': true/false, 'score': 0.0/0.5/1.0, 'explanation': 'brief explanation', 'unsupported_claims': ['list any unsupported claims']}}"""

    try:
        response = judge_llm.invoke(prompt).content
        result = parse_llm_json_response(response)
        score = result.get('score', 1.0 if result.get('faithful', False) else 0.0)
        
        return {
            'faithfulness_score': score,
            'explanation': result.get('explanation', ''),
            'unsupported_claims': result.get('unsupported_claims', [])
        }
    except Exception as e:
        print(f'Error evaluating faithfulness: {e}')
        return {'faithfulness_score': 0.0, 'explanation': 'Evaluation failed', 'unsupported_claims': []}


def evaluate_answer_relevance(query, answer, judge_llm):
    """
    nswer Relevance (A|Q): Does the answer address the question?
    
    rubric:
    - highly relevant (1.0): Answer directly and completely addresses the query
    - partially relevant (0.5): Answer addresses the query but misses important aspects
    - not relevant (0.0): Answer does not address the query
    
    args:
        query: User query
        answer: Generated answer
        judge_llm: LLM judge instance
        
    returns:
        Dict with answer relevance score and details
    """
    prompt = f"""You are evaluating whether an answer properly addresses a user query.

        RUBRIC:
        - Score 1.0 (Highly Relevant): Answer directly and completely addresses all aspects of the query
        - Score 0.5 (Partially Relevant): Answer addresses the query but misses important aspects or is incomplete
        - Score 0.0 (Not Relevant): Answer does not address the query or is completely off-topic

        Query: {query}

        Answer: {answer}

        Evaluate the relevance of the answer to the query.
        Respond with ONLY a JSON object:
        {{'relevant': true/false, 'score': 0.0/0.5/1.0, 'completeness_score': 0-10, 'explanation': 'brief explanation'}}"""

    try:
        response = judge_llm.invoke(prompt).content
        result = parse_llm_json_response(response)
        score = result.get('score', 1.0 if result.get('relevant', False) else 0.0)
        
        return {
            'answer_relevance_score': score,
            'completeness_score': result.get('completeness_score', 0) / 10.0,
            'explanation': result.get('explanation', '')
        }
    except Exception as e:
        print(f'Error evaluating answer relevance: {e}')
        return {'answer_relevance_score': 0.0, 'completeness_score': 0.0, 'explanation': 'Evaluation failed'}


def evaluate_tier2_generation(query, answer, contexts, judge_llm):
    """
    run all Tier 2 generation quality metrics.
    
    args:
        query: User query
        answer: Generated answer
        contexts: List of context strings
        judge_llm: LLM judge instance
        
    returns:
        Combined metrics dict
    """
    metrics = {}
    
    # Context Relevance (C|Q)
    context_rel = evaluate_context_relevance(query, contexts, judge_llm)
    metrics.update(context_rel)
    
    # Faithfulness (A|C)
    faithfulness = evaluate_faithfulness(answer, contexts, judge_llm)
    metrics.update(faithfulness)
    
    # Answer Relevance (A|Q)
    answer_rel = evaluate_answer_relevance(query, answer, judge_llm)
    metrics.update(answer_rel)
    
    return metrics

# TIER 3: ADVANCED QUALITY METRICS

def evaluate_context_support(contexts, answer, judge_llm):
    """
    Context Support (C|A): Do contexts sufficiently support generating the answer?
    
    rubric:
    - Fully Sufficient (1.0): Contexts contain all information needed to generate the answer
    - Partially Sufficient (0.5): Contexts provide most information, some details may be missing
    - Insufficient (0.0): Contexts lack critical information needed for the answer
    
    args:
        contexts: List of context strings
        answer: Generated answer
        judge_llm: LLM judge instance
        
    returns:
        Dict with context support score and details
    """
    context_combined = '\n\n'.join(contexts)
    
    prompt = f"""You are evaluating whether contexts provide sufficient information to generate an answer.

        RUBRIC:
        - Score 1.0 (Fully Sufficient): Contexts contain all necessary information to generate the complete answer
        - Score 0.5 (Partially Sufficient): Contexts provide most information, but some details may be missing or require minor inference
        - Score 0.0 (Insufficient): Contexts lack critical information needed to generate the answer

        Contexts:
        {context_combined}

        Answer:
        {answer}

        Evaluate whether the contexts sufficiently support generating this answer.
        Respond with ONLY a JSON object:
        {{'sufficient': true/false, 'score': 0.0/0.5/1.0, 'sufficiency_score': 0-10, 'explanation': 'brief explanation', 'missing_info': ['list any missing information']}}"""

    try:
        response = judge_llm.invoke(prompt).content
        result = parse_llm_json_response(response)
        score = result.get('score', 1.0 if result.get('sufficient', False) else 0.0)
        
        return {
            'context_support_score': score,
            'sufficiency_score': result.get('sufficiency_score', 0) / 10.0,
            'explanation': result.get('explanation', ''),
            'missing_info': result.get('missing_info', [])
        }
    except Exception as e:
        print(f'Error evaluating context support: {e}')
        return {'context_support_score': 0.0, 'sufficiency_score': 0.0, 'explanation': 'Evaluation failed', 'missing_info': []}


def evaluate_question_answerability(query, contexts, judge_llm):
    """
    Question Answerability (Q|C): Can the question be answered given the contexts?
    
    rubric:
    - fully answerable (1.0): Question can be completely answered using the contexts
    - partially answerable (0.5): Question can be partially answered, some aspects missing
    - not answerable (0.0): Question cannot be answered using the contexts
    
    Args:
        query: User query
        contexts: List of context strings
        judge_llm: LLM judge instance
        
    returns:
        Dict with answerability score and details
    """
    context_combined = '\n\n'.join(contexts)
    
    prompt = f"""You are evaluating whether a question can be answered using provided contexts.

        RUBRIC:
        - Score 1.0 (Fully Answerable): Question can be completely and accurately answered using only these contexts
        - Score 0.5 (Partially Answerable): Question can be partially answered, but some aspects cannot be addressed from contexts
        - Score 0.0 (Not Answerable): Question cannot be answered using these contexts

        Question: {query}

        Contexts:
        {context_combined}

        Evaluate whether this question can be answered using these contexts.
        Respond with ONLY a JSON object:
        {{'answerable': true/false, 'score': 0.0/0.5/1.0, 'confidence_score': 0-10, 'explanation': 'brief explanation'}}"""

    try:
        response = judge_llm.invoke(prompt).content
        result = parse_llm_json_response(response)
        score = result.get('score', 1.0 if result.get('answerable', False) else 0.0)
        
        return {
            'question_answerability_score': score,
            'confidence_score': result.get('confidence_score', 0) / 10.0,
            'explanation': result.get('explanation', '')
        }
    except Exception as e:
        print(f'Error evaluating question answerability: {e}')
        return {'question_answerability_score': 0.0, 'confidence_score': 0.0, 'explanation': 'Evaluation failed'}


def evaluate_self_containment(query, answer, judge_llm):
    """
    Self-Containment (Q|A): Does the answer make sense without the question?
    
    rubric:
    - Fully Self-Contained (1.0): Answer is clear and understandable without the question
    - Partially Self-Contained (0.5): Answer makes some sense but needs question for full context
    - Not Self-Contained (0.0): Answer is unclear without the question
    
    args:
        query: User query (for reference)
        answer: Generated answer
        judge_llm: LLM judge instance
        
    returns: Dict with self-containment score and details
    """
    prompt = f"""You are evaluating whether an answer is self-contained and understandable without its question.

        RUBRIC:
        - Score 1.0 (Fully Self-Contained): Answer is clear, complete, and understandable on its own without needing the question
        - Score 0.5 (Partially Self-Contained): Answer makes sense but would benefit from seeing the question for full clarity
        - Score 0.0 (Not Self-Contained): Answer is unclear or ambiguous without knowing the question

        Question (for reference): {query}

        Answer: {answer}

        Evaluate whether this answer is self-contained.
        Respond with ONLY a JSON object:
        {{'self_contained': true/false, 'score': 0.0/0.5/1.0, 'clarity_score': 0-10, 'explanation': 'brief explanation'}}"""

    try:
        response = judge_llm.invoke(prompt).content
        result = parse_llm_json_response(response)
        score = result.get('score', 1.0 if result.get('self_contained', False) else 0.0)
        
        return {
            'self_containment_score': score,
            'clarity_score': result.get('clarity_score', 0) / 10.0,
            'explanation': result.get('explanation', '')
        }
    except Exception as e:
        print(f'Error evaluating self-containment: {e}')
        return {'self_containment_score': 0.0, 'clarity_score': 0.0, 
                'explanation': 'Evaluation failed'}


def evaluate_tier3_advanced(query, answer, contexts, judge_llm):
    """
    run all Tier 3 advanced quality metrics.
    
    args:
        query: User query
        answer: Generated answer
        contexts: List of context strings
        judge_llm: LLM judge instance
        
    returns:
        Combined metrics dict
    """
    metrics = {}
    
    # context support (C|A)
    context_support = evaluate_context_support(contexts, answer, judge_llm)
    metrics.update(context_support)
    
    # question answerability (Q|C)
    question_ans = evaluate_question_answerability(query, contexts, judge_llm)
    metrics.update(question_ans)
    
    # self-containment (Q|A)
    self_contain = evaluate_self_containment(query, answer, judge_llm)
    metrics.update(self_contain)
    
    return metrics

# AGGREGATE METRICS

def compute_aggregate_metrics(all_results):
    """
    compute mean and std across all query results.
    
    args: all_results: List of evaluation result dicts
        
    returns: Dict with aggregate statistics per tier
    """
    aggregate = {
        'tier1_retrieval': defaultdict(list),
        'tier2_generation': defaultdict(list),
        'tier3_advanced': defaultdict(list)
    }
    
    # all metric values
    for result in all_results:
        for metric, value in result['tier1_retrieval'].items():
            if isinstance(value, (int, float)):
                aggregate['tier1_retrieval'][metric].append(value)
        
        for metric, value in result['tier2_generation'].items():
            if isinstance(value, (int, float)):
                aggregate['tier2_generation'][metric].append(value)
        
        for metric, value in result['tier3_advanced'].items():
            if isinstance(value, (int, float)):
                aggregate['tier3_advanced'][metric].append(value)
    
    # computing means and stds
    final_aggregate = {
        'tier1_retrieval': {},
        'tier2_generation': {},
        'tier3_advanced': {}
    }
    
    for tier in ['tier1_retrieval', 'tier2_generation', 'tier3_advanced']:
        for metric, values in aggregate[tier].items():
            if values:
                final_aggregate[tier][f'{metric}_mean'] = float(np.mean(values))
                final_aggregate[tier][f'{metric}_std'] = float(np.std(values))
    
    return final_aggregate