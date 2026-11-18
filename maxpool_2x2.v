// Modul Max-Pooling 2x2 untuk Reduksi Dimensi Feature Map
module maxpool_2x2 (
    input  wire               clk,
    input  wire               rst,
    input  wire               en,
    input  wire signed [31:0] in0, in1, in2, in3,
    output reg  signed [31:0] max_out,
    output reg                valid_out
);

    wire signed [31:0] max_01 = (in0 > in1) ? in0 : in1;
    wire signed [31:0] max_23 = (in2 > in3) ? in2 : in3;
    wire signed [31:0] max_all = (max_01 > max_23) ? max_01 : max_23;

    always @(posedge clk) begin
        if (rst) begin
            max_out   <= 32'sd0;
            valid_out <= 1'b0;
        end else if (en) begin
            max_out   <= max_all;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
