from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from scipy.spatial.distance import cosine, pdist, squareform
import time

class BERTSemanticDiversity:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model {model_name} loaded on {self.device}")
    
    def get_word_embedding(self, word, pooling_method='mean'):
        """
        使用BERT获取词嵌入
        
        Args:
            word: 输入词语
            pooling_method: 池化方法 ('mean', 'cls', 'max')
        """
        try:
            inputs = self.tokenizer(
                word, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=128
            )
            
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 获取最后一层的隐藏状态
            last_hidden_state = outputs.last_hidden_state
            
            # 应用不同的池化策略
            if pooling_method == 'mean':
                # 平均池化（排除padding tokens）
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = last_hidden_state * attention_mask
                embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
            elif pooling_method == 'cls':
                # 使用[CLS] token
                embedding = last_hidden_state[:, 0, :]
            elif pooling_method == 'max':
                # 最大池化
                embedding = torch.max(last_hidden_state, dim=1)[0]
            else:
                raise ValueError("pooling_method must be 'mean', 'cls', or 'max'")
            
            return embedding.cpu().numpy().squeeze()
            
        except Exception as e:
            print(f"Error processing word '{word}': {e}")
            return None
    
    def get_batch_embeddings(self, words, batch_size=32, pooling_method='mean'):
        """
        批量获取词嵌入，提高效率
        """
        embeddings = []
        valid_words = []
        
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i + batch_size]
            
            try:
                # 批量tokenize
                inputs = self.tokenizer(
                    batch_words, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True,
                    max_length=128
                )
                
                # 移动到设备
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                last_hidden_state = outputs.last_hidden_state
                
                # 批量池化
                if pooling_method == 'mean':
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    masked_hidden = last_hidden_state * attention_mask
                    batch_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
                elif pooling_method == 'cls':
                    batch_embeddings = last_hidden_state[:, 0, :]
                elif pooling_method == 'max':
                    batch_embeddings = torch.max(last_hidden_state, dim=1)[0]
                
                # 移动到CPU并转换为numpy
                batch_embeddings = batch_embeddings.cpu().numpy()
                
                embeddings.extend(batch_embeddings)
                valid_words.extend(batch_words)
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                # 如果批量失败，尝试逐个处理
                for word in batch_words:
                    emb = self.get_word_embedding(word, pooling_method)
                    if emb is not None:
                        embeddings.append(emb)
                        valid_words.append(word)
        
        return embeddings, valid_words
    
    def calculate_diversity(self, words, batch_size=32, pooling_method='mean'):
        """
        计算词语集的语义多样性
        
        Args:
            words: 词语列表
            batch_size: 批量处理大小
            pooling_method: 池化方法
        """
        start_time = time.time()
        
        # 批量获取词向量
        embeddings, valid_words = self.get_batch_embeddings(
            words, batch_size, pooling_method
        )
        
        if len(embeddings) < 2:
            print("Not enough valid words for diversity calculation")
            return None
        
        # 计算距离矩阵
        dist_matrix = squareform(pdist(embeddings, metric='cosine'))
        
        # 获取上三角矩阵（不包括对角线）
        upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        
        calculation_time = time.time() - start_time
        
        return {
            'words': valid_words,
            'distance_matrix': dist_matrix,
            'mean_distance': np.mean(upper_triangle),
            'std_distance': np.std(upper_triangle),
            'max_distance': np.max(upper_triangle),
            'min_distance': np.min(upper_triangle),
            'calculation_time': calculation_time,
            'num_words': len(valid_words),
            'embedding_dim': embeddings[0].shape[0]
        }
    
    def compare_word_pairs(self, word_pairs, pooling_method='mean'):
        """
        直接比较词语对的相似度
        """
        results = []
        
        for word1, word2 in word_pairs:
            emb1 = self.get_word_embedding(word1, pooling_method)
            emb2 = self.get_word_embedding(word2, pooling_method)
            
            if emb1 is not None and emb2 is not None:
                similarity = 1 - cosine(emb1, emb2)
                results.append({
                    'word1': word1,
                    'word2': word2,
                    'similarity': similarity,
                    'distance': 1 - similarity
                })
            else:
                results.append({
                    'word1': word1,
                    'word2': word2,
                    'similarity': None,
                    'distance': None,
                    'error': 'Word not found or processing error'
                })
        
        return results

# 使用示例
if __name__ == "__main__":
    # 初始化（自动选择GPU如果可用）
    bert_sd = BERTSemanticDiversity()
    
    # 测试数据
    # words = ['happy', 'joyful', 'cheerful', 'delighted', 'glad']
    # words = ['computer', 'ocean', 'philosophy', 'pizza', 'quantum']
    words = ['snow falling', 'deep snow', 'snow-covered houses', 'snowy sky']
    words = ['opening', 'caption', 'year', '1633']
    words = ['protagonist', 'man with gun', 'approaching', 'draws sword', 'cuts off two fingers']
    words = ['nude figure', 'snowy forest', 'holding blue sword', 'determined look', 'taking a bath']
    words = ['man runs', 'woman', 'holds books', 'drops books']
    words = ['Rock climbing', 'Southern California', 'Stoney Point Park', 'climbing gear', 'shirtless man', 'woman climbing', 'ropes', 'carabiners']
    
    print(words)
    print("计算相似词语的语义多样性:")
    result1 = bert_sd.calculate_diversity(words)
    if result1:
        print(f"平均距离: {result1['mean_distance']:.4f}")
        print(f"计算时间: {result1['calculation_time']:.2f}秒")
        print(f"处理词语数: {result1['num_words']}")
    