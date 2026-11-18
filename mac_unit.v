// Modul Multiply-Accumulate (MAC) 3x3 untuk Konvolusi CNN 8-bit Signed
module mac_unit (
    input  wire               clk,
    input  wire               rst,
    input  wire               en,
    input  wire signed [7:0]  p0, p1, p2, p3, p4, p5, p6, p7, p8,
    input  wire signed [7:0]  w0, w1, w2, w3, w4, w5, w6, w7, w8,
    input  wire signed [15:0] bias,
    output reg  signed [31:0] mac_out,
    output reg                valid_out
);

    wire signed [15:0] m0 = p0 * w0;
    wire signed [15:0] m1 = p1 * w1;
    wire signed [15:0] m2 = p2 * w2;
    wire signed [15:0] m3 = p3 * w3;
    wire signed [15:0] m4 = p4 * w4;
    wire signed [15:0] m5 = p5 * w5;
    wire signed [15:0] m6 = p6 * w6;
    wire signed [15:0] m7 = p7 * w7;
    wire signed [15:0] m8 = p8 * w8;

    always @(posedge clk) begin
        if (rst) begin
            mac_out   <= 32'sd0;
            valid_out <= 1'b0;
        end else if (en) begin
            mac_out   <= m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8 + {{16{bias[15]}}, bias};
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
