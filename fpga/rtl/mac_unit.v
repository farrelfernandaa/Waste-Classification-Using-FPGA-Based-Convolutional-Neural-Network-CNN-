`timescale 1ns / 1ps

module mac_unit (
    input wire clk,
    input wire rst,
    input wire en,

    input wire signed [7:0] p0, p1, p2, p3, p4, p5, p6, p7, p8,
    input wire signed [7:0] w0, w1, w2, w3, w4, w5, w6, w7, w8,

    input wire signed [15:0] bias,

    output reg signed [31:0] mac_out,
    output reg valid_out
);

    // Signed multiplication: 8-bit x 8-bit = 16-bit
    wire signed [15:0] m0 = p0 * w0;
    wire signed [15:0] m1 = p1 * w1;
    wire signed [15:0] m2 = p2 * w2;
    wire signed [15:0] m3 = p3 * w3;
    wire signed [15:0] m4 = p4 * w4;
    wire signed [15:0] m5 = p5 * w5;
    wire signed [15:0] m6 = p6 * w6;
    wire signed [15:0] m7 = p7 * w7;
    wire signed [15:0] m8 = p8 * w8;

    // Explicit sign extension to 32-bit
    wire signed [31:0] m0_ext = {{16{m0[15]}}, m0};
    wire signed [31:0] m1_ext = {{16{m1[15]}}, m1};
    wire signed [31:0] m2_ext = {{16{m2[15]}}, m2};
    wire signed [31:0] m3_ext = {{16{m3[15]}}, m3};
    wire signed [31:0] m4_ext = {{16{m4[15]}}, m4};
    wire signed [31:0] m5_ext = {{16{m5[15]}}, m5};
    wire signed [31:0] m6_ext = {{16{m6[15]}}, m6};
    wire signed [31:0] m7_ext = {{16{m7[15]}}, m7};
    wire signed [31:0] m8_ext = {{16{m8[15]}}, m8};

    wire signed [31:0] bias_ext =
        {{16{bias[15]}}, bias};

    // 3x3 convolution sum
    wire signed [31:0] sum =
        m0_ext + m1_ext + m2_ext +
        m3_ext + m4_ext + m5_ext +
        m6_ext + m7_ext + m8_ext +
        bias_ext;

    always @(posedge clk) begin
        if (rst) begin
            mac_out  <= 32'sd0;
            valid_out <= 1'b0;
        end
        else begin
            valid_out <= 1'b0;

            if (en) begin
                mac_out  <= sum;
                valid_out <= 1'b1;
            end
        end
    end

endmodule
