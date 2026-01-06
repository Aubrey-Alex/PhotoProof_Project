import hashlib
import json
import numpy as np

class PhotoProvenance:
    """
    【数据溯源模块】(Mock Implementation)
    模拟 PhotoProof 中的密码学组件：相机签名与递归证明聚合。
    这里使用哈希和结构化数据来模拟，而非真实的 RSA/SNARKs。
    """
    
    @staticmethod
    def _compute_image_hash(image):
        """计算图像内容的哈希指纹"""
        # 将 numpy 数组序列化并计算 SHA256
        data = image.tobytes()
        return hashlib.sha256(data).hexdigest()[:16] # 取前16位作为简略指纹

    @staticmethod
    def mint_camera_signature(image, device_id="Cam-001"):
        """
        [模拟相机] 生成原始媒体的数字签名
        """
        img_hash = PhotoProvenance._compute_image_hash(image)
        signature = {
            "type": "camera_attestation",
            "device_id": device_id,
            "content_hash": img_hash,
            "timestamp": "2024-01-01T12:00:00Z" # 模拟时间戳
        }
        return signature

    @staticmethod
    def generate_proof_package(original_sig, operation_log, final_image):
        """
        [模拟 Prover] 生成最终的证明包
        包含：原始签名 + 操作历史 + 最终内容指纹
        """
        final_hash = PhotoProvenance._compute_image_hash(final_image)
        
        proof_package = {
            "root_of_trust": original_sig,
            "transformations": operation_log,
            "final_claim": {
                "hash": final_hash,
                "status": "derived_honestly"
            }
        }
        return proof_package

    @staticmethod
    def verify_provenance(proof_package, final_image_check):
        """
        [模拟 Verifier] 验证数据链完整性
        """
        # 1. 验证最终图像是否匹配
        current_hash = PhotoProvenance._compute_image_hash(final_image_check)
        claimed_hash = proof_package["final_claim"]["hash"]
        
        if current_hash != claimed_hash:
            return False, "❌ 最终图像哈希不匹配 (被篡改)"
            
        # 2. 检查信任根 (是否有相机签名)
        if proof_package["root_of_trust"]["type"] != "camera_attestation":
            return False, "❌ 缺失相机原始签名"
            
        return True, "✅ 信任链完整 (Camera -> Transform -> Final)"
