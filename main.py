import numpy as np
from PIL import Image, ImageEnhance
import os
import cv2
import random
from core.universal_verifier import UniversalVerifier
from core.provenance import PhotoProvenance # 引入溯源模块
from core.circuit_visualizer import CircuitVisualizer # 引入电路可视化模块

# 配置
VIDEO_PATH = "data/original/test_video.mp4"
OUTPUT_DIR = "demo_output"
SAMPLE_INTERVAL = 30 

def run_full_stack_demo():
    print("="*60)
    print("PhotoProof Pro: 全栈视频真实性验证系统")
    print("   1. Cryptographic Provenance (模拟签名与信任链)")
    print("   2. Arithmetic Constraints (数学逻辑验证)")
    print("   3. Circuit Visualization (底层电路生成)")
    print("============================================================")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 请先在 data 文件夹下放入 test_video.mp4")
        return
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    circuits_dir = os.path.join(OUTPUT_DIR, "circuits")
    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    if not os.path.exists(circuits_dir): os.makedirs(circuits_dir)
    if not os.path.exists(frames_dir): os.makedirs(frames_dir)

    cap = cv2.VideoCapture(VIDEO_PATH)
    verifier = UniversalVerifier()
    
    frame_idx = 0
    HAS_GENERATED_CIRCUITS = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > 60:
            break
            
        if frame_idx % SAMPLE_INTERVAL == 0:
            print(f"\n\n--- [关键帧 ID: {frame_idx}] 验证流水线启动 ---")
            
            # ... (Existing Code) ...
            # 原始帧 (BGR -> RGB)
            original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = original.shape
            
            # [Step A: 溯源] 模拟相机签名
            print(f"📷 [Provenance] 相机生成原始签名 (Trust Root)...")
            camera_sig = PhotoProvenance.mint_camera_signature(original)
            ops_log = [] # 记录操作日志

            # --- [Visualization Trigger] ---
            if not HAS_GENERATED_CIRCUITS:
                print("\n[System] 正在编译算术电路可视化图表 (Output to demo_output/)...")
                
                # 1. Brightness Circuit
                cv_brit = CircuitVisualizer("Brightness_Logic")
                cv_brit.build_brightness_circuit(1.0, 50.0)
                cv_brit.render(circuits_dir)
                
                # 2. Crop Circuit
                cv_crop = CircuitVisualizer("Crop_Logic")
                cv_crop.build_crop_circuit(100, 100) # 示例参数
                cv_crop.render(circuits_dir)
                
                # 3. Rotation Circuit
                cv_rot = CircuitVisualizer("Rotation_Logic")
                cv_rot.build_paeth_rotation_circuit(15.0)
                cv_rot.render(circuits_dir)
                
                HAS_GENERATED_CIRCUITS = True
                print("[System] 电路图生成完毕，继续执行动态验证...\n")

            # ==========================================
            # 变换 1: 亮度调节
            # ==========================================
            print("1️⃣  执行变换: 亮度 +50...")
            alpha, beta = 1.0, 50.0
            bright_frame = cv2.convertScaleAbs(original, alpha=alpha, beta=beta)
            ops_log.append({"op": "brightness", "params": {"alpha": alpha, "beta": beta}})
            
            # 验证亮度
            check_points = [(w//2, h//2), (10, 10), (w-10, h-10)] 
            pass_count = 0
            for cx, cy in check_points:
                val_in = int(original[cy, cx][1])
                val_out = int(bright_frame[cy, cx][1])
                is_valid, err = verifier.verify_brightness(val_in, val_out, alpha, beta)
                if is_valid: pass_count += 1
            
            if pass_count == len(check_points):
                print(f"   ✅ [Math] 亮度线性约束检查通过")
            else:
                print(f"   ❌ [Math] 亮度验证失败")

            # ==========================================
            # 变换 2: 中心裁剪
            # ==========================================
            print("2️⃣  执行变换: 中心裁剪 400x400...")
            crop_w, crop_h = 400, 400
            start_x = (w - crop_w) // 2
            start_y = (h - crop_h) // 2
            cropped_frame = bright_frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
            ops_log.append({"op": "crop", "params": {"x": start_x, "y": start_y, "w": crop_w, "h": crop_h}})
            
            # 验证裁剪
            p_out_test = (0, 0)
            p_in_test = (start_x, start_y)
            is_mapped, _ = verifier.verify_crop(p_in_test, p_out_test, (start_x, start_y, crop_w, crop_h))
            pixel_match = np.array_equal(cropped_frame[0,0], bright_frame[start_y, start_x])
            
            if is_mapped and pixel_match:
                print(f"   ✅ [Math] 裁剪空间映射检查通过")

            # ==========================================
            # 变换 3: 旋转 15 度
            # ==========================================
            print("3️⃣  执行变换: 旋转 15 度...")
            img_pil = Image.fromarray(cropped_frame)
            rotated_pil = img_pil.rotate(15, resample=Image.BICUBIC)
            rotated_final = np.array(rotated_pil)
            ops_log.append({"op": "rotate", "params": {"angle": 15}})
            
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx}_final.jpg"), cv2.cvtColor(rotated_final, cv2.COLOR_RGB2BGR))
                        # 验证旋转 (Probabilistic)
            print("   Start Probabilistic Verification (Samples=50)...")
            is_rot_valid, score = verifier.verify_paeth_rotation_probabilistic(cropped_frame, rotated_final, 15.0, samples=50)
            
            if is_rot_valid:
                print(f"   ✅ [Math] 旋转蒙特卡洛验证通过 (Confidence: {score*100:.1f}%)")
            else:
                print(f"   ❌ [Math] 旋转验证失败 (Confidence: {score*100:.1f}%)")

            # [Step B: 最终验签]
            print(f"🔒 [Finalize] 验证完整证据链...")
            proof_pkg = PhotoProvenance.generate_proof_package(camera_sig, ops_log, rotated_final)
            is_proven, msg = PhotoProvenance.verify_provenance(proof_pkg, rotated_final)
            print(f"   {msg}")

            print(f"🎉 关键帧 {frame_idx} 验证完成！")

        frame_idx += 1

    cap.release()
    print("\n所有演示结束。")

if __name__ == "__main__":
    run_full_stack_demo()