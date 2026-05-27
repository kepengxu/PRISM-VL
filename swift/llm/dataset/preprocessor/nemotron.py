"""Nemotron 数据集预处理器"""
from typing import Any, Dict, Optional
from swift.llm.dataset.preprocessor.core import MessagesPreprocessor


class NemotronPreprocessor(MessagesPreprocessor):
    """处理 Nemotron 数据集的 input/output 格式"""
    
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 将 input/output 格式转换为 messages 格式
        if 'input' in row and 'output' in row:
            messages = row['input']  # input 已经是标准 messages 格式
            messages.append({
                'role': 'assistant',
                'content': row['output']
            })
            row['messages'] = messages
            row.pop('input', None)
            row.pop('output', None)
        
        # 调用父类方法继续处理
        return super().preprocess(row)
