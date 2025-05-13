import torch
import json
import re
import os
import traceback
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

class QueryAnalyzer:
    """处理图文共查模式下的查询分析"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get("model_path", "/data/hzj/projects/multi-modal-retrieval/baseline/MMGenerativeIR/Llama-3.2-11B-Vision-Instruct")
        self.max_attempts = config.get("max_attempts", 3)
        self.temperature = config.get("temperature", 0.3)
        self.max_new_tokens = config.get("max_new_tokens", 1024)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
        
    def _initialize_model(self):
        """初始化多模态大模型"""
        print(f"正在加载查询分析模型: {self.model_path}")
        try:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.processor.tokenizer.padding_side = 'left'
            print("查询分析模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            traceback.print_exc()
            raise
    
    def process_analysis(self, answer: str) -> str:
        """处理和清理模型返回的JSON响应"""
        try:
            # 步骤1: 提取核心JSON内容
            match = re.search(
                r'(?i)(?:assistant|answer)[\s\S]*?({[\s\S]*})',
                answer,
                re.DOTALL
            )
            
            if not match:
                return "JSON_NOT_FOUND"

            # 步骤2: 基本清理
            cleaned = match.group(1)
            cleaned = re.sub(r'\\"', '"', cleaned)  # 处理转义引号
            cleaned = re.sub(r'\n', ' ', cleaned)   # 替换换行符
            
            # 步骤3: 修复常见格式错误
            # 修复缺失的引号
            cleaned = re.sub(
                r'([{,]\s*)([a-zA-Z_]+)(\s*:)',
                lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}',
                cleaned
            )
            # 删除末尾逗号
            cleaned = re.sub(r',\s*(}|])', r'\1', cleaned)
            # 将单引号转换为双引号
            cleaned = re.sub(r"'", '"', cleaned)
            
            # 步骤4: 验证JSON
            try:
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError as e:
                # 修复缺失的右括号
                open_braces = cleaned.count('{') - cleaned.count('}')
                if open_braces > 0:
                    cleaned += '}' * open_braces
                return cleaned
                
        except Exception as e:
            return f"PROCESSING_ERROR: {str(e)}"

    def analyze_query(self, image: Image.Image, query_text: str) -> Dict[str, Any]:
        """分析查询文本和相关图像，提取关键词和增强查询"""
        attempt = 0
        while attempt < self.max_attempts:
            try:
                # 准备提示词
                image_path = getattr(image, 'filename', 'uploaded_image')
                messages = {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"""You are a multimodal semantic parser. Analyze the user's query and associated image to generate structured retrieval keywords by following these steps:

                1. Cross-Modal Entity Recognition:
                - Identify explicit entities (nouns/verbs)
                - Resolve pronouns like \"it\" or \"this\" using image context
                - REQUIRED: Extract at least 1 implicit keyword from image context

                2. Generate Valid JSON (STRICT FORMAT):
                {{
                  \"original_query\": \"[EXACT_USER_QUERY]\",
                  \"explicit_keywords\": [\"term1\", \"term2\"],
                  \"implicit_keywords\": [\"MUST_HAVE_AT_LEAST_1_ITEM\"],  // REQUIRED FIELD
                  \"augmented_query\": \"Natural fusion of visual and textual clues\"
                }}

                Bad Example (REJECT): 
                {{"implicit_keywords": []}}  // EMPTY ARRAY NOT ALLOWED
                {{"implicit_keywords": "object"}}  // STRING INSTEAD OF ARRAY

                Good Example:
                Image: park bench with pigeons 
                Query: \"Why are they gathered here?\"
                Output:
                {{
                  \"original_query\": \"Why are they gathered here?\",
                  \"explicit_keywords\": [\"gathered\"],
                  \"implicit_keywords\": [\"pigeons\", \"park bench\", \"feeding\"],
                  \"augmented_query\": \"Why are pigeons gathered around this park bench for feeding?\"
                }}

                Current Task:
                Image: {image_path}
                Query: \"{query_text}\"

                Generate valid JSON (IMPLICIT_KEYWORDS MUST BE NON-EMPTY ARRAY):""".replace("{", "{{").replace("}", "}}")}
                    ]
                }
                input_text = self.processor.apply_chat_template([messages], add_generation_prompt=True)

                # 处理输入并动态填充
                inputs = self.processor(
                    images=[[image]], 
                    text=[input_text], 
                    return_tensors="pt", 
                    add_special_tokens=True, 
                    padding="longest",  # 优化不同长度输入
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)

                # 生成响应
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature, 
                )
                
                generated_response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

                # 增强响应处理
                processed_json = self.process_analysis(generated_response)
                parsed_json = json.loads(processed_json)

                # 校验数据结构
                required_keys = ["original_query", "explicit_keywords", "implicit_keywords", "augmented_query"]
                for key in required_keys:
                    if key not in parsed_json:
                        raise KeyError(f"Missing required key: {key}")
                    
                if not isinstance(parsed_json["implicit_keywords"], list):
                    raise TypeError(f"implicit_keywords must be list, got {type(parsed_json['implicit_keywords'])}")
                    
                if len(parsed_json["implicit_keywords"]) == 0:
                    raise ValueError("implicit_keywords array is empty (minimum 1 item required)")

                return {
                    "success": True,
                    "raw": generated_response,
                    "processed": processed_json,
                    "analysis": parsed_json
                }

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:  
                attempt += 1
                error_detail = f"{type(e).__name__}: {str(e)}"
                print(f"Attempt {attempt} failed for query '{query_text}' - {error_detail}")
                
                if attempt == self.max_attempts:
                    return {
                        "success": False,
                        "raw": generated_response if 'generated_response' in locals() else None,
                        "processed": processed_json if 'processed_json' in locals() else None,
                        "error": f"Max attempts reached. Last error: {error_detail}",
                    }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Critical error: {str(e)}",
                    "traceback": traceback.format_exc()
                }
                
        return {"success": False, "error": "Failed to analyze query after multiple attempts."}
    
    def get_enhanced_query(self, image: Image.Image, query_text: str) -> Tuple[str, Dict[str, Any]]:
        """分析查询并返回增强的查询文本和分析结果"""
        result = self.analyze_query(image, query_text)
        
        if result["success"]:
            analysis = result["analysis"]
            enhanced_query = analysis["augmented_query"]
            return enhanced_query, result
        else:
            # 如果分析失败，返回原始查询
            print(f"查询分析失败，使用原始查询: {result.get('error', 'Unknown error')}")
            return query_text, result