"""Optional text-distance ranking extracted from the supplied BDD experiment."""

def sort_candidates_by_distance(data_dict, model_name='openai/clip-vit-base-patch32'):
    """
    给定 ID 和候选词的字典，计算相似度并按到 ID 集合的距离排序。
    
    参数:
    data_dict: dict, 格式例如 {"cat": ["clouded leopard", "caracal", ...], "car": ["golf cart", ...]}
    model_name: str, 默认使用 CLIP 模型提取文本特征
    """
    import torch
    from transformers import CLIPProcessor, CLIPTextModel
    if not data_dict or not all(isinstance(v, list) for v in data_dict.values()):
        raise ValueError('Expected a nonempty mapping of ID names to candidate lists')
    print(f'Loading text encoder ({model_name})...')
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    id_labels = list(data_dict.keys())
    all_candidates = set()
    for candidates in data_dict.values():
        all_candidates.update(candidates)
    candidate_list = sorted(all_candidates - set(id_labels))
    if not candidate_list:
        raise ValueError("No candidate labels remain after excluding ID labels")
    print(f'Extracted {len(id_labels)} ID labels and {len(candidate_list)} unique candidates.')
    with torch.no_grad():
        id_inputs = processor(text=id_labels, return_tensors='pt', padding=True)
        id_inputs = {k: v.to(model.device) for k, v in id_inputs.items()}
        id_embeds = model(**id_inputs).pooler_output
        id_embeds = id_embeds / id_embeds.norm(p=2, dim=-1, keepdim=True)
        cand_inputs = processor(text=candidate_list, return_tensors='pt', padding=True)
        cand_inputs = {k: v.to(model.device) for k, v in cand_inputs.items()}
        cand_embeds = model(**cand_inputs).pooler_output
        cand_embeds = cand_embeds / cand_embeds.norm(p=2, dim=-1, keepdim=True)
    similarity_matrix = torch.matmul(cand_embeds, id_embeds.T)
    max_sim_values, max_sim_indices = torch.max(similarity_matrix, dim=1)
    distances = 1.0 - max_sim_values
    results = []
    for i, candidate in enumerate(candidate_list):
        closest_id = id_labels[max_sim_indices[i].item()]
        sim_score = max_sim_values[i].item()
        dist_score = distances[i].item()
        results.append({'candidate': candidate, 'closest_id': closest_id, 'max_sim': round(sim_score, 4), 'dist': round(dist_score, 4)})
    sorted_results = sorted(results, key=lambda x: x['dist'], reverse=True)
    return sorted_results
