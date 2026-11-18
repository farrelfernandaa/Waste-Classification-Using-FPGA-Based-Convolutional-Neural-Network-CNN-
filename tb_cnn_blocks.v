`timescale 1ns / 1ps

module tb_cnn_blocks;
    reg clk;
    reg rst;
    reg en;

    reg signed [7:0] p0, p1, p2, p3, p4, p5, p6, p7, p8;
    reg signed [7:0] w0, w1, w2, w3, w4, w5, w6, w7, w8;
    reg signed [15:0] bias;

    wire signed [31:0] mac_out;
    wire mac_valid;
    wire signed [31:0] relu_out;

    mac_unit uut_mac (
        .clk(clk), .rst(rst), .en(en),
        .p0(p0), .p1(p1), .p2(p2),
        .p3(p3), .p4(p4), .p5(p5),
        .p6(p6), .p7(p7), .p8(p8),
        .w0(w0), .w1(w1), .w2(w2),
        .w3(w3), .w4(w4), .w5(w5),
        .w6(w6), .w7(w7), .w8(w8),
        .bias(bias),
        .mac_out(mac_out),
        .valid_out(mac_valid)
    );

    relu_activation uut_relu (
        .data_in(mac_out),
        .data_out(relu_out)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1; en = 0;
        #20 rst = 0;

        // Uji 1: Hasil konvolusi positif
        @(posedge clk);
        en = 1;
        p0 = 8'sd10; p1 = 8'sd20; p2 = 8'sd10;
        p3 = 8'sd5;  p4 = 8'sd15; p5 = 8'sd5;
        p6 = 8'sd10; p7 = 8'sd20; p8 = 8'sd10;

        w0 = 8'sd1;  w1 = 8'sd2;  w2 = 8'sd1;
        w3 = 8'sd0;  w4 = 8'sd1;  w5 = 8'sd0;
        w6 = -8'sd1; w7 = -8'sd2; w8 = -8'sd1;
        bias = 16'sd5;

        @(posedge clk);
        en = 0;
        #10;
        $display("Uji 1 -> MAC Out: %d | ReLU Out: %d", mac_out, relu_out);

        // Uji 2: Hasil konvolusi negatif (harus dipotong jadi 0 oleh ReLU)
        @(posedge clk);
        en = 1;
        bias = -16'sd100;

        @(posedge clk);
        en = 0;
        #10;
        $display("Uji 2 -> MAC Out: %d | ReLU Out: %d", mac_out, relu_out);

        #20 $finish;
    end
endmodule
