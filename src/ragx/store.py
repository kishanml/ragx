    
chunk_entities_extraction_prompt = """
Extract high-quality knowledge from the text.

## INPUT
{text_chunk}

## TASK
1. Write a concise, information-dense summary.
2. Generate 5-7 meaningful questions answerable strictly from the text.
3. Extract 5-7 important keywords/phrases capturing complete meaning.

## RULES
- Cover key facts, concepts, entities, and relationships.
- Questions must be specific, diverse, and non-duplicate.
- Keywords should be meaningful phrases, not isolated words.
- Do not hallucinate or use external knowledge.
- Prefer quality over quantity.
"""