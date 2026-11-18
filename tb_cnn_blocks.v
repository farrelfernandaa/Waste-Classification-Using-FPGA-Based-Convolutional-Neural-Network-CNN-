`timescale 1ns / 1ps

module tb_cnn_blocks;

    reg clk;
    reg rst;
    reg en;

    reg signed [7:0] p0, p1, p2, p3, p4, p5, p6, p7, p8;
    reg signed [7:0] w0, w1, w2, w3, w4, w5, w6, w7, w8;
    reg signed [15:0] bias;

    wire signed [31:0] mac_out;
    wire valid_out;

    wire signed [31:0] relu_out;

    integer test_count;
    integer pass_count;
    integer fail_count;

    mac_unit uut_mac (
        .clk(clk),
        .rst(rst),
        .en(en),

        .p0(p0), .p1(p1), .p2(p2),
        .p3(p3), .p4(p4), .p5(p5),
        .p6(p6), .p7(p7), .p8(p8),

        .w0(w0), .w1(w1), .w2(w2),
        .w3(w3), .w4(w4), .w5(w5),
        .w6(w6), .w7(w7), .w8(w8),

        .bias(bias),

        .mac_out(mac_out),
        .valid_out(valid_out)
    );

    relu_activation uut_relu (
        .data_in(mac_out),
        .data_out(relu_out)
    );

    // Clock: 10 ns period
    always #5 clk = ~clk;

    // --------------------------------------------------
    // Task: check MAC output
    // --------------------------------------------------
    task check_result;
        input signed [31:0] expected;
        input [8*40-1:0] test_name;

        begin
            test_count = test_count + 1;

            if (mac_out === expected && valid_out === 1'b1) begin
                pass_count = pass_count + 1;

                $display(
                    "PASS: %0s | MAC=%0d | Expected=%0d",
                    test_name,
                    mac_out,
                    expected
                );
            end
            else begin
                fail_count = fail_count + 1;

                $display(
                    "FAIL: %0s | MAC=%0d | Expected=%0d | Valid=%b",
                    test_name,
                    mac_out,
                    expected,
                    valid_out
                );
            end
        end
    endtask

    // --------------------------------------------------
    // Test sequence
    // --------------------------------------------------
    initial begin

        clk = 0;
        rst = 1;
        en = 0;

        test_count = 0;
        pass_count = 0;
        fail_count = 0;

        // Initialize inputs
        p0 = 0; p1 = 0; p2 = 0;
        p3 = 0; p4 = 0; p5 = 0;
        p6 = 0; p7 = 0; p8 = 0;

        w0 = 0; w1 = 0; w2 = 0;
        w3 = 0; w4 = 0; w5 = 0;
        w6 = 0; w7 = 0; w8 = 0;

        bias = 0;

        // Reset
        @(negedge clk);
        rst = 0;

        // ==================================================
        // TEST 1: Positive multiplication
        // 9 x (1 x 1) + 3 = 12
        // ==================================================

        @(negedge clk);

        en = 1;

        p0 = 1; p1 = 1; p2 = 1;
        p3 = 1; p4 = 1; p5 = 1;
        p6 = 1; p7 = 1; p8 = 1;

        w0 = 1; w1 = 1; w2 = 1;
        w3 = 1; w4 = 1; w5 = 1;
        w6 = 1; w7 = 1; w8 = 1;

        bias = 3;

        @(posedge clk);
        #1;

        check_result(32'sd12, "Positive MAC");

        // ==================================================
        // TEST 2: Negative multiplication
        // 9 x (-2 x 3) + 0 = -54
        // ==================================================

        @(negedge clk);

        p0 = -2; p1 = -2; p2 = -2;
        p3 = -2; p4 = -2; p5 = -2;
        p6 = -2; p7 = -2; p8 = -2;

        w0 = 3; w1 = 3; w2 = 3;
        w3 = 3; w4 = 3; w5 = 3;
        w6 = 3; w7 = 3; w8 = 3;

        bias = 0;

        @(posedge clk);
        #1;

        check_result(-32'sd54, "Negative MAC");

        // ==================================================
        // TEST 3: Mixed signed multiplication
        // 1*2 + 2*3 + 3*(-1) +
        // 4*2 + 5*(-2) + 6*1 +
        // 7*(-1) + 8*2 + 9*(-1) + 10
        // = 24
        // ==================================================

        @(negedge clk);

        p0 = 1; p1 = 2; p2 = 3;
        p3 = 4; p4 = 5; p5 = 6;
        p6 = 7; p7 = 8; p8 = 9;

        w0 = 2; w1 = 3; w2 = -1;
        w3 = 2; w4 = -2; w5 = 1;
        w6 = -1; w7 = 2; w8 = -1;

        bias = 10;

        @(posedge clk);
        #1;

        check_result(32'sd24, "Mixed Signed MAC");

        // ==================================================
        // TEST 4: Maximum signed multiplication
        // 9 x (-128 x -128) = 147456
        // ==================================================

        @(negedge clk);

        p0 = -128; p1 = -128; p2 = -128;
        p3 = -128; p4 = -128; p5 = -128;
        p6 = -128; p7 = -128; p8 = -128;

        w0 = -128; w1 = -128; w2 = -128;
        w3 = -128; w4 = -128; w5 = -128;
        w6 = -128; w7 = -128; w8 = -128;

        bias = 0;

        @(posedge clk);
        #1;

        check_result(32'sd147456, "Maximum Signed MAC");

        // ==================================================
        // TEST 5: ReLU negative output
        // ==================================================

        @(negedge clk);

        p0 = -2; p1 = -2; p2 = -2;
        p3 = -2; p4 = -2; p5 = -2;
        p6 = -2; p7 = -2; p8 = -2;

        w0 = 3; w1 = 3; w2 = 3;
        w3 = 3; w4 = 3; w5 = 3;
        w6 = 3; w7 = 3; w8 = 3;

        bias = 0;

        @(posedge clk);
        #1;

        check_result(-32'sd54, "ReLU Negative Input");

        if (relu_out !== 32'sd0) begin
            fail_count = fail_count + 1;
            $display("FAIL: ReLU | Output=%0d | Expected=0", relu_out);
        end
        else begin
            pass_count = pass_count + 1;
            $display("PASS: ReLU | Output=%0d", relu_out);
        end

        // ==================================================
        // TEST 6: ReLU positive output
        // ==================================================

        @(negedge clk);

        p0 = 1; p1 = 1; p2 = 1;
        p3 = 1; p4 = 1; p5 = 1;
        p6 = 1; p7 = 1; p8 = 1;

        w0 = 1; w1 = 1; w2 = 1;
        w3 = 1; w4 = 1; w5 = 1;
        w6 = 1; w7 = 1; w8 = 1;

        bias = 3;

        @(posedge clk);
        #1;

        check_result(32'sd12, "ReLU Positive Input");

        if (relu_out !== 32'sd12) begin
            fail_count = fail_count + 1;
            $display("FAIL: ReLU | Output=%0d | Expected=12", relu_out);
        end
        else begin
            pass_count = pass_count + 1;
            $display("PASS: ReLU | Output=%0d", relu_out);
        end

        // ==================================================
        // TEST 7: Enable disabled
        // Output must hold previous value
        // ==================================================

        @(negedge clk);

        en = 0;

        @(posedge clk);
        #1;

        test_count = test_count + 1;

        if (valid_out === 1'b0 && mac_out === 32'sd12) begin
            pass_count = pass_count + 1;
            $display("PASS: Enable Disabled");
        end
        else begin
            fail_count = fail_count + 1;
            $display("FAIL: Enable Disabled");
        end

        // ==================================================
        // FINAL SUMMARY
        // ==================================================

        #10;

        $display("-----------------------------------------");
        $display("TEST SUMMARY");
        $display("-----------------------------------------");
        $display("Total Tests : %0d", test_count);
        $display("Passed      : %0d", pass_count);
        $display("Failed      : %0d", fail_count);
        $display("-----------------------------------------");

        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        $finish;

    end

endmodule
