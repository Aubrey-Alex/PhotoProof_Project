import graphviz
import os

class CircuitVisualizer:
    """
    负责将抽象的数学约束转化为可视化的电路图。
    构建计算图 (Computation Graph)。
    """
    def __init__(self, transform_type):
        self.name = f"{transform_type}_Circuit"
        # 创建有向图，设置从左到右布局
        self.dot = graphviz.Digraph(comment=self.name, format='png')
        self.dot.attr(rankdir='LR', bgcolor='white')
        self.node_count = 0

    def add_node(self, label, shape='circle', color='black', style='solid', fontcolor='black'):
        node_id = f"n_{self.node_count}"
        self.dot.node(node_id, label, shape=shape, color=color, style=style, fontcolor=fontcolor)
        self.node_count += 1
        return node_id

    def add_gate(self, input_nodes, operation, output_label, color='orange'):
        """添加一个逻辑门 (Gate)"""
        # 门节点
        gate_id = f"gate_{self.node_count}"
        self.node_count += 1
        # 门的样式更像芯片
        self.dot.node(gate_id, operation, shape='note', style='filled', fillcolor=color, fontcolor='white')
        
        # 输出变量节点 (中间 Witness)
        out_id = self.add_node(output_label, shape='ellipse', color='gray', style='dashed')
        
        # 连接输入 -> 门
        for inp in input_nodes:
            self.dot.edge(inp, gate_id, arrowsize='0.5')
        # 连接门 -> 输出
        self.dot.edge(gate_id, out_id, arrowsize='0.5')
        return out_id

    def build_paeth_rotation_circuit(self, angle):
        """
        构建论文提到的 Paeth 剪切旋转电路 (3次 Shear)
        展示 X 和 Y 信号在三次剪切中的交互流程。
        """
        self.dot.attr(label=f"Arithmetic Circuit: Paeth Rotation Constrains (Angle={angle})", labelloc='t', fontsize='20')
        
        # 定义信号流
        # 输入
        x_in = self.add_node("X_in", shape='doublecircle', color='blue', fontcolor='blue')
        y_in = self.add_node("Y_in", shape='doublecircle', color='blue', fontcolor='blue')
        
        # 常量
        alpha = self.add_node("Const: tan(a/2)", shape='box', style='filled', color='lightgrey')
        beta = self.add_node("Const: sin(a)", shape='box', style='filled', color='lightgrey')
        
        # --- Shear 1 (X changes) ---
        # x1 = x + y * alpha
        # y1 = y
        with self.dot.subgraph(name='cluster_shear1') as c:
            c.attr(label='Shear Step 1 (X-Shear)', style='dashed', color='blue')
            mul1 = self.add_gate([y_in, alpha], "MUL", "y*alpha")
            x1 = self.add_gate([x_in, mul1], "ADD", "X_1 (Witness)")
            y1 = y_in # Y 不变，直接连过去
            
        # --- Shear 2 (Y changes) ---
        # y2 = y1 + x1 * beta
        # x2 = x1
        with self.dot.subgraph(name='cluster_shear2') as c:
            c.attr(label='Shear Step 2 (Y-Shear)', style='dashed', color='green')
            mul2 = self.add_gate([x1, beta], "MUL", "x1*beta")
            y2 = self.add_gate([y1, mul2], "ADD", "Y_2 (Witness)")
            x2 = x1 # X 不变
            
        # --- Shear 3 (X changes) ---
        # x3 = x2 + y2 * alpha
        # y3 = y2
        with self.dot.subgraph(name='cluster_shear3') as c:
            c.attr(label='Shear Step 3 (X-Shear)', style='dashed', color='blue')
            mul3 = self.add_gate([y2, alpha], "MUL", "y2*alpha")
            x3 = self.add_gate([x2, mul3], "ADD", "X_Final")
            y3 = y2
            
        # 输出
        x_out = self.add_node("X_Out", shape='doublecircle', color='red', fontcolor='red')
        y_out = self.add_node("Y_Out", shape='doublecircle', color='red', fontcolor='red')
        
        self.dot.edge(x3, x_out, label='Constraint check')
        self.dot.edge(y3, y_out, label='Constraint check')

    def build_brightness_circuit(self, alpha, beta):
        """构建亮度调节电路 + 范围检查 (Range Proof)"""
        self.dot.attr(label=f"Arithmetic Circuit: Brightness & Range Proof", labelloc='t', fontsize='20')
        
        # 模拟 3 个通道并行处理
        for i, channel in enumerate(['R', 'G', 'B']):
            with self.dot.subgraph(name=f'cluster_{channel}') as c:
                c.attr(label=f'Channel {channel} Processor', style='filled', color='#eeeeee')
                
                px_in = self.add_node(f"{channel}_In", shape='circle')
                
                # 线性变换
                linear = self.add_gate([px_in], f"Linear(x*{alpha}+{beta})", f"{channel}_Transformed")
                
                # 范围证明 (Range Proof Gadget)
                # 这是一个这也是 R1CS 中很重要的部分：证明 0 <= x <= 255
                # 通常通过分解为 8 个 bit 来证明
                range_check = self.add_gate([linear], "RangeCheck(8-bit)", f"{channel}_Bits", color='purple')
                
                px_out = self.add_node(f"{channel}_Out", shape='doublecircle')
                self.dot.edge(range_check, px_out)

    def build_crop_circuit(self, x_off, y_off):
        """构建空间映射验证电路"""
        self.dot.attr(label=f"Spatial Mapping Circuit (Crop: {x_off},{y_off})", labelloc='t', fontsize='20')
        
        with self.dot.subgraph(name='cluster_mapping') as c:
            c.attr(label='Coordinate Mapping Verification', color='purple')
            
            # 两个输入：声明的 crop 参数 和 像素索引
            idx_node = self.add_node("Pixel_Index (i, j)", shape='invtrapezium')
            param_node = self.add_node(f"Offset ({x_off}, {y_off})", shape='box')
            
            # 加法门
            addr_calc = self.add_gate([idx_node, param_node], "ADDR_ADD", "Src_Addr_Calc")
            
            # 内存查找 (Lookup)
            mem_node = self.add_node("Merkle_Tree_Root\n(Original_Image_Commitment)", shape='cylinder', color='gold')
            lookup = self.add_gate([addr_calc, mem_node], "Merkle_Lookup", "Src_Pixel_Value", color='gold')
            
            target_pixel = self.add_node("Target_Pixel_Value", shape='doublecircle')
            
            # 相等约束
            eq_check = self.add_gate([lookup, target_pixel], "EQ_ASSERT", "Valid_Bit", color='red')

    def render(self, output_dir="demo_output"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        path = os.path.join(output_dir, f"{self.name}")
        # 这一步会生成 .png 图片
        try:
            output_path = self.dot.render(path, cleanup=True)
            print(f"   📊 [Visualizer] 电路图已生成: {output_path}")
            return output_path
        except Exception as e:
            print(f"⚠️ [Visualizer] Graphviz 未检测到，已生成原始 dot 文件: {path}.dot")
            # 如果没有 graphviz，保存源码
            with open(path + ".dot", "w") as f:
                f.write(self.dot.source)
            return path + ".dot"