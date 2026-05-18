
import re
import traceback
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import DoclingDocument
from litellm import query

from faiss_vec import FaissVectorDB
from schemas import Chunk, Document, Output
from store import chunk_entities_extraction_prompt
from tokenizer import Tokenizer
from utils import Timer, generate_sha256_hash, preprocess_text


from dotenv import load_dotenv
load_dotenv()

class DocSegmenter(Tokenizer):
    
    
    HEADING_PATTERN: re.Pattern[str] = re.compile(r"##(.*?)(?=\n)")

    
    def __init__(self, model_name : str = "gpt-5.4-mini"):
        
        self._llm  = ChatOpenAI(model=model_name, temperature=0)
        super().__init__()
        
    
    def extract_content(self, filepath : str, * , device : str = 'cpu') -> str:
        
        fp = Path(filepath)
        try:    
            _parser = DocumentConverter()
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options.device = device
            _parser = DocumentConverter(format_options={
                            "pdf": PdfFormatOption(
                                pipeline_options=pipeline_options,
                            )
                        })       
            
            if fp.exists():
                
                parsed_doc: DoclingDocument = _parser.convert(filepath).document
                md = parsed_doc.export_to_markdown()
                                
                return Document(id=generate_sha256_hash(md), name = parsed_doc.name, page_count=parsed_doc.num_pages(), source=md)
                
            else:
                raise FileNotFoundError(f"{fp} path doesn't exist ! Please provide a valid document path to extract.")    
            
        except Exception:
            print(f'Error occured while extracting content : \n{traceback.format_exc()}')
            
    
    def _title_based_chunking(self, markdown : str) -> tuple[list, list, list]:
        
        matches = list(DocSegmenter.HEADING_PATTERN.finditer(markdown))
        
        if not matches:
            return [], [], []
        
        text_chunks, text_tokens, text_topics = [],[],[]
        total_length = len(markdown)
        
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i + 1].start() if i + 1 < len(matches) else total_length

            chunk = markdown[start:end].replace("\n\n", "\n")
            split_chunk = chunk.split(" ")
            
            text_chunks.append(split_chunk)
            text_tokens.append([self.count_tokens(ch) for ch in split_chunk])
            text_topics.append(matches[i].group(0))
            
        return text_chunks, text_tokens, text_topics     
    
       
    
    def _adjust_token_size(self, text_chunks : list[list[int]], text_tokens: list[list[int]], text_topics : list[str], max_token : int =512, overlap_tokens : int = 50):
        
        final_chunks = []
        current_tokens = 0
        current_chunk_parts = []
        
        max_token-= overlap_tokens

        for (words, w_token, topic) in zip(text_chunks, text_tokens, text_topics):
            
            text_ = " ".join(words)
            chunk_text = text_
            token = sum(w_token)
            topic = re.sub(r"#+\s",repl="",string=topic)
            
            if  token > max_token:
                
                index, context = self.get_token_context(w_token, words, max_token)
                        
                overlapping_context =""
                if overlap_tokens:
                    _,overlapping_context = self.get_token_context(w_token[index:], words[index:], min(abs(len(w_token) - index), overlap_tokens))
                    context_str = context + overlapping_context
                    cleaned_context_str = preprocess_text(context_str)
                    
                final_chunks.append(Chunk(id = generate_sha256_hash(context_str), title=topic, source=context_str, cleaned_text=cleaned_context_str))
                
                current_chunk_parts = [topic+"\n"+ " ".join(words[index:])]
                current_tokens: int = sum(w_token[index:])
                
            elif current_tokens + token <= max_token:     
                current_chunk_parts.append(chunk_text)
                current_tokens += token
            
            else:
                
                overlapping_context =""
                if overlap_tokens:
                    index ,overlapping_context = self.get_token_context(w_token, words, overlap_tokens)
                    
                context_str = "".join(current_chunk_parts) + overlapping_context
                cleaned_context_str = preprocess_text(context_str)

                final_chunks.append(Chunk(id=generate_sha256_hash(context_str), title=topic, source=context_str))        
                current_chunk_parts = [chunk_text]
                current_tokens = token
                
        if current_chunk_parts:
            context_str = "".join(current_chunk_parts) + overlapping_context
            cleaned_context_str = preprocess_text(context_str)

            final_chunks.append(Chunk(id= generate_sha256_hash(context_str),title=topic, source=context_str))
        return final_chunks    
    

    
    def _update_chunk_entities(self, chunk  : Chunk):
        
        try:
            response = self._llm.with_structured_output(Output).invoke([SystemMessage(content=chunk_entities_extraction_prompt.format(text_chunk= chunk.source))])
            chunk.summary = response.summary
            chunk.questions = response.questions
            chunk.keywords = response.keywords

            return chunk
            
        except Exception:
            print(f'Error occured while generating chunk entities : \n{traceback.format_exc()}')

            
            
    def generate_chunks(self, document_path : str, *, document : Document = None):
        
        if document is None:
            with Timer('Extracting document content'):
                document = self.extract_content(document_path)
        with Timer("Chunking document"):
            text_chunks, text_tokens, text_topics = self._title_based_chunking(document.source)
            assert len(text_chunks) == len(text_tokens) == len(text_topics)        
            chunk_list = self._adjust_token_size(text_chunks, text_tokens, text_topics)
        
        with Timer("Generating chunk's entities"):
            document.chunks = list(map(self._update_chunk_entities, chunk_list))
        
        return document
    
            
    
            
    
        
        

if __name__ == "__main__":
    
    document_fp = Path("/home/kishanm/Documents/ragx/document/CTMP Process Brightness Optimisation and Control_ Latency Tank to Bleach Tower.pdf")
    
    if not document_fp.exists():
        print('yes')
    doc_seg = DocSegmenter()
    vec_db = FaissVectorDB()
    data = doc_seg.generate_chunks(document_path=document_fp)
    db = vec_db.store(data.chunks)
    print(db.query(query_text="what is ctmp ?",k=5))
    
        
    
        
            
    
        
    