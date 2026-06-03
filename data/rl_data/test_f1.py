from sentence_transformers import SentenceTransformer, util
import torch

def calculate_semantic_precision_recall_gpu(prediction_str, reference_str, threshold=0.7, model_name='all-MiniLM-L6-v2'):
    """
    GPU优化的语义Precision和Recall计算函数。
    """

    # 1. 预处理
    pred_elements = [elem.strip() for elem in prediction_str.split(',') if elem.strip()]
    ref_elements = [elem.strip() for elem in reference_str.split(',') if elem.strip()]

    # 处理空值情况
    if not pred_elements and not ref_elements:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'true_positives': 0, 'pred_count': 0, 'ref_count': 0}
    elif not pred_elements:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'true_positives': 0, 'pred_count': 0, 'ref_count': len(ref_elements)}
    elif not ref_elements:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'true_positives': 0, 'pred_count': len(pred_elements), 'ref_count': 0}

    # 2. 加载模型并确保在GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name).to(device)
    
    # 3. 编码所有元素（直接在GPU上生成tensor）
    pred_embeddings = model.encode(pred_elements, convert_to_tensor=True, device=device)
    ref_embeddings = model.encode(ref_elements, convert_to_tensor=True, device=device)

    # 4. 计算余弦相似度矩阵（全部在GPU上完成）
    cosine_scores = util.cos_sim(pred_embeddings, ref_embeddings)  # shape: (pred_count, ref_count)

    # 5. 在GPU上找到每个预测元素最相似的参考元素
    best_scores, best_indices = torch.max(cosine_scores, dim=1)  # 在dim=1上取max，即对每个预测元素找最佳参考元素

    # 6. 创建匹配掩码（Mask）
    # 条件1: 相似度超过阈值
    above_threshold = best_scores > threshold
    # 条件2: 确保每个参考元素只被匹配一次
    matched_ref_mask = torch.zeros(len(ref_elements), dtype=torch.bool, device=device)
    
    true_positives = 0
    # 按照相似度分数从高到低排序，优先匹配分数高的对
    sorted_indices = torch.argsort(best_scores, descending=True)
    
    for idx in sorted_indices:
        if above_threshold[idx]:
            ref_idx = best_indices[idx]
            if not matched_ref_mask[ref_idx]:
                # 这个参考元素还没被匹配过，可以匹配
                true_positives += 1
                matched_ref_mask[ref_idx] = True

    # 7. 计算指标
    pred_count = len(pred_elements)
    ref_count = len(ref_elements)

    precision = true_positives / pred_count
    recall = true_positives / ref_count
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision.item() if torch.is_tensor(precision) else precision,
        'recall': recall.item() if torch.is_tensor(recall) else recall,
        'f1': f1.item() if torch.is_tensor(f1) else f1,
        'true_positives': true_positives,
        'pred_count': pred_count,
        'ref_count': ref_count,
        'device_used': str(device)
    }

# 示例用法
if __name__ == "__main__":
    # 测试数据
    model_prediction = "chicken breast, broccoli, garlic, soy sauce, olive oil, salt"
    ground_truth = "chicken, fresh broccoli, minced garlic, light soy, extra virgin oil, black pepper, sea salt"

    results = calculate_semantic_precision_recall_gpu(model_prediction, ground_truth, threshold=0.7)

    print(f"计算设备: {results['device_used']}")
    print(f"模型预测: {model_prediction}")
    print(f"参考答案: {ground_truth}")
    print("\n评估结果:")
    print(f"精确率 (Precision): {results['precision']:.4f}")
    print(f"召回率 (Recall): {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"匹配元素数 (TP): {results['true_positives']}")
    print(f"预测元素总数: {results['pred_count']}")
    print(f"参考元素总数: {results['ref_count']}")