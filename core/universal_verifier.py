import numpy as np
import math

class UniversalVerifier:
    """
    【核心算术约束系统】
    对应论文 V-E 节：实现多种变换的 R1CS 数学约束检查。
    """
    
    # --- 1. 旋转约束 (原本的 Paeth 算法) ---
    def __init__(self):
        pass

    def verify_paeth_rotation(self, p_in, p_out, angle_deg, center):
        """验证旋转 (基于 3-Shear 约束)"""
        angle_rad = math.radians(-angle_deg)
        alpha = -math.tan(angle_rad / 2)
        beta = math.sin(angle_rad)
        
        cx, cy = center
        x, y = p_in[0] - cx, p_in[1] - cy
        
        # 3次 Shear 计算
        x1 = x + y * alpha
        y1 = y
        x2 = x1
        y2 = y1 + x1 * beta
        x3 = x2 + y2 * alpha
        y3 = y2
        
        # 校验
        final_x, final_y = x3 + cx, y3 + cy
        diff = abs(final_x - p_out[0]) + abs(final_y - p_out[1])
        return diff < 1.5, diff

    # --- 2. 亮度/对比度约束 (新增) ---
    def verify_brightness(self, val_in, val_out, alpha, beta):
        """
        对应论文 V-E: Contrast/Brightness
        约束公式: val_out = clip(val_in * alpha + beta, 0, 255)
        """
        # 计算理论值
        theoretical = val_in * alpha + beta
        
        # 处理边界截断 (Clipping Constraint)
        # 如果理论值 > 255 且 实际值 == 255，则视为通过 (因为被截断了)
        # 如果理论值 < 0 且 实际值 == 0，则视为通过
        if theoretical > 255:
            is_valid = (val_out == 255)
        elif theoretical < 0:
            is_valid = (val_out == 0)
        else:
            # 正常范围内，允许 1.0 的浮点误差
            is_valid = abs(theoretical - val_out) < 2.0
            
        return is_valid, abs(theoretical - val_out)

    # --- 3. 裁剪约束 (新增) ---
    def verify_crop(self, p_in_coord, p_out_coord, crop_params):
        """
        对应论文 V-E: Crop
        约束公式: P_out(x, y) == P_in(x + x_start, y + y_start)
        """
        x_start, y_start, w, h = crop_params
        
        # 1. 边界约束检查 (Boundary Check)
        if p_out_coord[0] >= w or p_out_coord[1] >= h:
             return False, 999 # 超出裁剪框
        
        # 2. 坐标映射约束 (Coordinate Mapping)
        expected_x = p_out_coord[0] + x_start
        expected_y = p_out_coord[1] + y_start
        
        # 检查映射关系是否严格成立
        is_mapped = (expected_x == p_in_coord[0]) and (expected_y == p_in_coord[1])
        return is_mapped, 0 if is_mapped else 1

    def verify_paeth_rotation_probabilistic(self, img_in, img_out, angle, samples=50):
        """
        【蒙特卡洛验证】
        不验证全图 200万个像素，而是随机抽取 50 个点进行数学轨迹检查。
        如果这 50 个点都符合 Paeth 旋转公式，则认为整图可信。
        """
        import random
        h, w = img_in.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        pass_count = 0
        valid_samples = 0
        
        # 预计算参数
        angle_rad = math.radians(-angle)
        alpha = -math.tan(angle_rad / 2)
        beta = math.sin(angle_rad)
        
        # 随机生成测试点
        test_points = [(random.randint(0, w-1), random.randint(0, h-1)) for _ in range(samples)]
        
        for px, py in test_points:
            # Paeth 坐标变换计算
            x = px - center_x
            y = py - center_y
            
            x1 = x + y * alpha
            y2 = y + x1 * beta # y1 = y
            x3 = x1 + y2 * alpha # x2 = x1
            
            tx = x3 + center_x
            ty = y2 + center_y # y3 = y2
            
            # 取整获取像素坐标
            itx, ity = int(round(tx)), int(round(ty))
            
            if 0 <= itx < w and 0 <= ity < h:
                valid_samples += 1
                color_in = img_in[py, px].astype(float)
                color_out = img_out[ity, itx].astype(float)
                
                # 比较颜色一致性
                diff = np.mean(np.abs(color_in - color_out))
                if diff < 35.0: # 允许插值误差
                    pass_count += 1
                    
        if valid_samples == 0: return False, 0.0
        score = pass_count / valid_samples
        return score > 0.8, score