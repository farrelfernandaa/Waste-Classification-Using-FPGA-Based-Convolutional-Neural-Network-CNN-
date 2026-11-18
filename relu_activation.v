// Modul Fungsi Aktivasi ReLU (Rectified Linear Unit)
module relu_activation (
    input  wire signed [31:0] data_in,
    output wire signed [31:0] data_out
);
    // Jika bit tanda (MSB) bernilai 1 (negatif), keluaran menjadi 0
    assign data_out = (data_in[31] == 1'b1) ? 32'sd0 : data_in;

endmodule
