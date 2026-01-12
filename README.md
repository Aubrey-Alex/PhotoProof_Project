# PhotoProof Project: Image/Video Integrity Verification Demo

## 1. 项目简介 (Project Overview)

本项目是 **PhotoProof** 概念的演示实现。PhotoProof 是一种基于密码学的图像编辑验证系统，旨在解决数字时代的图像真实性问题。

本系统演示了一个完整的“全栈”验证流程：
1.  **可信溯源 (Provenance)**: 使用RSA密钥生成相机数字签名（信任根）。
2.  **图像变换 (Transformations)**: 对图像进行亮度调节 (Brightness)、裁剪 (Crop) 和旋转 (Rotation) 操作。
3.  **电路可视化 (Circuit Visualization)**: 自动生成上述操作对应的算术电路逻辑图 (Arithmetic Circuit)，展示底层验证逻辑。
4.  **数学验证 (Math/Mock Verification)**: 使用 Python 逻辑模拟验证者 (Verifier) 检查图像变换的合法性。

## 2. 与 PhotoProof 原论文的对比 (Comparison with Original Paper)

本项目旨在**可视化**和**演示** PhotoProof 的核心逻辑，而非提供生产级别的密码学工具。以下是本项目与原论文方案的主要区别：

### 2.1 缺失部分 (What is Missing)
*   **真实的零知识证明 (Real ZK-SNARKs)**:
    *   原论文使用 libsnark/circom 等后端生成真实的 `R1CS` 约束和 `zk-SNARK` 证明 (Proof)。
    *   本项目在 `main.py` 中使用 Python 逻辑 (Assertions & Probabilistic Checks) **模拟** 了验证过程，并未真正执行耗时的密码学证明生成 (Setup/Prove/Verify)。
    *   *注：`cpp_circuit_source/` 目录下保留了使用 libsnark 编写亮度 Gadget 的 C++ 源码片段，作为真实实现的参考。*
*   **零知识隐私性 (Zero-Knowledge Property)**:
    *   原论文中，Verifier **不可见** 原始图像，只能看到图像的承诺 (Commitment) 和证明。
    *   本 Demo 为了便于演示，Verifier 直接读取了原始像素数据来验证变换结果是否正确。
*   **复杂的密码学承诺 (Cryptographic Commitments)**:
    *   原论文使用 Pedersen Commitments 或 Merkle Tree Root 来锚定图像内容。
    *   本项目简化为使用 SHA-256 哈希值来模拟图像指纹。

### 2.2 所做更改与优化 (Modifications & Features)
*   **可视化增强**: 新增了 `CircuitVisualizer` 模块。原论文通常只展示数学公式，本项目利用 Graphviz 将抽象的数学约束（如 Paeth 旋转的三次剪切、亮度调节的范围检查）转化为直观的**流程图/电路图**，便于理解计算图结构。
*   **视频流支持**: 将原本针对单张图片的验证扩展到了**视频流** (Video Stream)。系统可以逐帧或抽样对视频内容进行“拍摄 -> 编辑 -> 验证”的完整流水线演示。
*   **交互式演示**: `main.py` 提供了一个控制台驱动的演示流程，实时打印每一步的操作日志 (Ops Log)、数学验证结果和置信度 (Confidence)，适合教学展示。
*   **RSA数字签名**: 增强了溯源模块，使用真实的RSA密钥进行数字签名而非简单哈希。首次运行时自动生成密钥对，提供更强的完整性和身份验证。
*   **自动输出组织**: 输出文件自动分类到 `circuits/` 和 `frames/` 子文件夹，保持项目结构清晰。

## 3. 功能模块 (Modules)

*   `main.py`: 项目入口，协调完整的演示流程。
*   `core/provenance.py`: **密码学组件**。使用RSA密钥生成相机签名、操作日志打包以及最终的数字签名验证。首次运行时自动生成密钥对并保存到 `keys/` 文件夹。
*   `core/circuit_visualizer.py`: **核心可视化组件**。将亮度、剪切、旋转等数学逻辑绘制成 `.dot` 和 `.png` 图表。
*   `core/universal_verifier.py`: 包含各个变换的数学验证逻辑（如亮度线性检查、旋转的蒙特卡洛采样检查）。
*   `cpp_circuit_source/`: 存放真实的 C++ 电路代码示例 (Reference only)。

## 4. 快速开始 (Quick Start)

### 环境要求
*   Python 3.8+
*   运行依赖库:
    ```bash
    pip install numpy opencv-python pillow graphviz cryptography
    ```
    *(注意: 生成图片需要系统安装 Graphviz 软件并配置环境变量)*

### 运行演示
1.  确保 `data/original/` 目录下存在测试视频 `test_video.mp4`。
2.  运行主程序：
    ```bash
    python main.py
    ```
3.  程序将输出：
    *   控制台日志：显示每一步的签名生成、操作执行和验证结果。
    *   `demo_output/circuits/` 目录：生成的算术电路结构图 (`.dot` 文件和源码文件)。
    *   `demo_output/frames/` 目录：处理后的视频帧 (`.jpg` 文件)。
    *   `keys/` 目录：首次运行时自动生成的RSA密钥文件 (`camera_secret.key`、`camera_public.key`、`verifier_public.key`)。
