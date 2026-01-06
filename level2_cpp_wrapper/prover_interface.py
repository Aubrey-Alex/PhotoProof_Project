import hashlib
import json

def generate_proof(img_in, img_out, transform_type, params, prev_proof):
    """
    模拟生成 PCD 证明 (Algorithm 3)
    在真实实现中，这里会运行 libsnark 生成 .proof 文件
    """
    
    # 计算图像哈希，用于锁定内容
    img_in_hash = hashlib.sha256(img_in.tobytes()).hexdigest()
    img_out_hash = hashlib.sha256(img_out.tobytes()).hexdigest()
    
    proof_packet = {
        "type": "PCD_PROOF",
        "transform": transform_type,
        "params": params,
        "input_hash": img_in_hash,
        "output_hash": img_out_hash,
        "prev_proof": prev_proof,  # 递归包含上一步的证明 (Recursive Composition)
        "is_valid": True # 模拟标记，实际由数学验证决定
    }
    
    return proof_packet