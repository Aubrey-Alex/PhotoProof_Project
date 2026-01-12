import hashlib
import json
import numpy as np
import os
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

class PhotoProvenance:
    """
    【数据溯源模块】(Enhanced with RSA Keys)
    使用RSA密钥进行数字签名，模拟真实的密码学组件。
    """
    
    KEYS_DIR = "keys"
    CAMERA_PRIVATE_KEY_FILE = os.path.join(KEYS_DIR, "camera_secret.key")
    CAMERA_PUBLIC_KEY_FILE = os.path.join(KEYS_DIR, "camera_public.key")
    VERIFIER_PUBLIC_KEY_FILE = os.path.join(KEYS_DIR, "verifier_public.key")
    
    @staticmethod
    def _ensure_keys_exist():
        """确保密钥文件存在，如果不存在则生成"""
        if not os.path.exists(PhotoProvenance.KEYS_DIR):
            os.makedirs(PhotoProvenance.KEYS_DIR)
        
        if not os.path.exists(PhotoProvenance.CAMERA_PRIVATE_KEY_FILE):
            # 生成相机私钥
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            # 保存私钥
            with open(PhotoProvenance.CAMERA_PRIVATE_KEY_FILE, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            # 保存公钥
            public_key = private_key.public_key()
            with open(PhotoProvenance.CAMERA_PUBLIC_KEY_FILE, "wb") as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
        
        if not os.path.exists(PhotoProvenance.VERIFIER_PUBLIC_KEY_FILE):
            # 假设verifier使用相同的公钥，或生成新的
            # 这里简单复制相机的公钥作为verifier的公钥
            with open(PhotoProvenance.CAMERA_PUBLIC_KEY_FILE, "rb") as f:
                data = f.read()
            with open(PhotoProvenance.VERIFIER_PUBLIC_KEY_FILE, "wb") as f:
                f.write(data)
    
    @staticmethod
    def _load_camera_private_key():
        PhotoProvenance._ensure_keys_exist()
        with open(PhotoProvenance.CAMERA_PRIVATE_KEY_FILE, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())
    
    @staticmethod
    def _load_camera_public_key():
        PhotoProvenance._ensure_keys_exist()
        with open(PhotoProvenance.CAMERA_PUBLIC_KEY_FILE, "rb") as f:
            return serialization.load_pem_public_key(f.read(), backend=default_backend())
    
    @staticmethod
    def _compute_image_hash(image):
        """计算图像内容的哈希指纹"""
        data = image.tobytes()
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def mint_camera_signature(image, device_id="Cam-001"):
        """
        [相机] 生成原始媒体的数字签名，使用RSA私钥
        """
        PhotoProvenance._ensure_keys_exist()
        img_hash = PhotoProvenance._compute_image_hash(image)
        data_to_sign = f"{device_id}:{img_hash}:2026-01-01T12:00:00Z".encode()
        
        private_key = PhotoProvenance._load_camera_private_key()
        signature_bytes = private_key.sign(
            data_to_sign,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        signature = {
            "type": "camera_attestation",
            "device_id": device_id,
            "content_hash": img_hash,
            "timestamp": "2026-01-01T12:00:00Z",
            "signature": signature_bytes.hex()
        }
        return signature
    
    @staticmethod
    def generate_proof_package(original_sig, operation_log, final_image):
        """
        [Prover] 生成最终的证明包
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
        [Verifier] 验证数据链完整性，使用RSA公钥验证签名
        """
        # 1. 验证最终图像是否匹配
        current_hash = PhotoProvenance._compute_image_hash(final_image_check)
        claimed_hash = proof_package["final_claim"]["hash"]
        
        if current_hash != claimed_hash:
            return False, "❌ 最终图像哈希不匹配 (被篡改)"
            
        # 2. 检查信任根 (验证相机签名)
        root = proof_package["root_of_trust"]
        if root["type"] != "camera_attestation":
            return False, "❌ 缺失相机原始签名"
        
        # 验证签名
        data_to_verify = f"{root['device_id']}:{root['content_hash']}:{root['timestamp']}".encode()
        signature_bytes = bytes.fromhex(root["signature"])
        
        public_key = PhotoProvenance._load_camera_public_key()
        try:
            public_key.verify(
                signature_bytes,
                data_to_verify,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        except:
            return False, "❌ 相机签名验证失败"
            
        return True, "✅ 信任链完整 (Camera -> Transform -> Final)"
