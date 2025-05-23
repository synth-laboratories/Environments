module adder4(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    assign sum = a + b;
endmodule

module adder4_tb;
    reg [3:0] a, b;
    wire [4:0] sum;
    
    adder4 dut(.a(a), .b(b), .sum(sum));
    
    initial begin
        a = 4'b0000; b = 4'b0000; #10;
        if (sum != 5'b00000) $fatal(1, "Test failed: 0 + 0 != 0");
        
        a = 4'b0001; b = 4'b0001; #10;
        if (sum != 5'b00010) $fatal(1, "Test failed: 1 + 1 != 2");
        
        a = 4'b1111; b = 4'b0001; #10;
        if (sum != 5'b10000) $fatal(1, "Test failed: 15 + 1 != 16");
        
        $display("ALL_TESTS_PASSED");
        $finish;
    end
endmodule