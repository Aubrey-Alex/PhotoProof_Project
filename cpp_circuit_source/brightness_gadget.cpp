/**
 * PhotoProof Native Circuit Implementation
 * Gadget: Brightness & Contrast
 * Paper Reference: Section V-E
 *
 * Implements R1CS constraint: |alpha * x + beta| = y
 */

#include <libsnark/gadgetlib1/gadget.hpp>
#include <libsnark/gadgetlib1/protoboard.hpp>

using namespace libsnark;

template<typename FieldT>
class brightness_gadget : public gadget<FieldT> {
private:
    pb_variable<FieldT> pixel_in;
    pb_variable<FieldT> pixel_out;
    pb_variable<FieldT> alpha;
    pb_variable<FieldT> beta;
    pb_variable<FieldT> intermediate; // 用于存储 alpha * pixel_in

public:
    // 构造函数：定义电路的输入输出线
    brightness_gadget(protoboard<FieldT> &pb,
                      const pb_variable<FieldT> &in,
                      const pb_variable<FieldT> &out,
                      const pb_variable<FieldT> &a,
                      const pb_variable<FieldT> &b) :
        gadget<FieldT>(pb, "brightness_gadget"), pixel_in(in), pixel_out(out), alpha(a), beta(b) {
            // 分配中间变量
            intermediate.allocate(pb, "intermediate_val");
        }

    // 核心：生成数学约束 (Constraints)
    void generate_r1cs_constraints() {
        // 1. 乘法约束: pixel_in * alpha = intermediate
        // add_r1cs_constraint(A, B, C) 意味着 A * B = C
        this->pb.add_r1cs_constraint(
            r1cs_constraint<FieldT>(pixel_in, alpha, intermediate),
            "pixel_in * alpha = intermediate"
        );

        // 2. 加法约束: intermediate + beta = pixel_out
        // 在 R1CS 中，加法是线性的，写成 (1 * (intermediate + beta) = pixel_out)
        this->pb.add_r1cs_constraint(
            r1cs_constraint<FieldT>(1, intermediate + beta, pixel_out),
            "intermediate + beta = pixel_out"
        );
        
        // 注意：真实场景还需添加比较器 Gadget (Comparison Gadget) 来处理 0-255 截断
        // 此处为核心线性变换逻辑演示
    }

    // 生成见证 (Witness)：即带入具体数值计算
    void generate_r1cs_witness() {
        FieldT a_val = this->pb.val(alpha);
        FieldT in_val = this->pb.val(pixel_in);
        FieldT b_val = this->pb.val(beta);

        this->pb.val(intermediate) = in_val * a_val;
        this->pb.val(pixel_out) = this->pb.val(intermediate) + b_val;
    }
};