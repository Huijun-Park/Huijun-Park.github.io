---
title:  "[FPGA / Verilog]Image Convolution With Xilinx PYNQ"
date:   2022-08-15 00:00:00 +0900
categories: jekyll update
layout: post
---

FPGA can be handy when making hardware systems used for specific computation loads at scale but with less flexibility.
Here, we are going to make a simple image convolution filter with a size of 5 x 5 pixels using a xilinx pynq board.

FPGA has very real and serious limitations that you have to consider.
1. Block RAM is very limited.
2. SRAM is even more limited.

This is basically why we have to think in terms of data pipelines with limited random access, similarly to GPU computing.
For the movement of data, we are going to utilize AXI4 Stream DMA which is a part of AMBA protocol.
With that in mind, let's analyze the 2D convolution filter.
A 2D convolution filter can be seen as an accumulation of 1D convolution filters.
A 1D convolution filter can be seen as an FIR filter.
For FPGAs there are tools specialized for DSP calculation such as FIR filters and in this case it is called DSP slices.
We are going to extensively utilize these for efficient computation.
<br>![DSP Chain](/assets/images/pynq_conv/dsp_chain.PNG)<br>
With a single chain we can calculate 1d convolution of an image row and the kernel.
With five of these we can calculate the 2d convolution by summing them.
{% highlight verilog %}
  //1clk
  adder_tree[0] <= row_acc[0] + row_acc[1];
  adder_tree[1] <= row_acc[2] + row_acc[3];
  adder_tree[2] <= row_acc[4];
  //2clk
  adder_tree[3] <= adder_tree[0]+adder_tree[1];
  adder_tree[4] <= adder_tree[2];
  //3clk
  adder_tree[5] <= adder_tree[3]+adder_tree[4];
{% endhighlight %}

To run five convolutions simutaneously, we need to store five recent image rows.
Five rows is not a huge data but it still is too large for SRAM so we will use BRAM.
A BRAM module's operation must meet specific requirements or it will be synthesized as SRAM. 
{% highlight verilog %}
module dual_port_ram #(
        parameter integer RAM_WIDTH = 32,
        parameter integer RAM_HEIGHT = 640
    ) (clka,clkb,ena,enb,wea,addra,addrb,dia,dob);
    input clka,clkb,ena,enb,wea;
    input [10:0] addra,addrb;
    input [RAM_WIDTH-1:0] dia;
    output reg [RAM_WIDTH-1:0] dob;
    (* ram_style = "block" *) reg[RAM_WIDTH-1:0] ram [RAM_HEIGHT-1:0];
    always @(posedge clka) begin if (ena)
    begin
     if (wea)
     ram[addra] <= dia;
     end
    end
    always @(posedge clkb) begin if (enb)
    begin
     dob <= ram[addrb];
     end
    end
endmodule
{% endhighlight %}

We instantiate not only the image buffer rows but also output row buffer to store the result before sending it.
{% highlight verilog %}
    dual_port_ram #(
        .RAM_WIDTH(32),
        .RAM_HEIGHT(COLUMNS)
    ) image_buffer[0:ROWS-1] (
        .clka(sclk),.clkb(mclk),.ena(1'b1),.enb(1'b1),
        .wea(image_buffer_we),.addra(image_buffer_wr_addr_wire),.addrb(image_buffer_rd_addr_wire),.dia(image_buffer_wr_data_wire),.dob(image_buffer_rd_data_wire)
    );

    dual_port_ram #(
        .RAM_WIDTH(32),
        .RAM_HEIGHT(COLUMNS)
    ) output_row (
        .clka(mclk),.clkb(mclk),.ena(1'b1),.enb(1'b1),
        .wea(output_row_we),.addra(output_row_wr_addr),.addrb(output_row_rd_addr),.dia(output_row_wr_data),.dob(output_row_rd_data_wire)
    );
{% endhighlight %}

For the 5x5 kernel, the size is very tolerable to be stored in SRAM. We can make a simple DMA input and output for checking.
{% highlight verilog %}
    // bufffer ready //
    always @(posedge kernel_sclk)begin
        if(kernel_s_axis_tvalid)begin
            kernel_s_axis_tready <= 1'b1;
        end
        else begin
            kernel_s_axis_tready <= 1'b0;
        end
    end
    
    reg [31:0] kernel_buffer[0:KERNEL_SIZE-1];
    
    // fill buffer //
    always @(posedge kernel_sclk)begin
        if(!kernel_s_axis_aresetn || (kernel_s_axis_tdata==32'hffff_ffff && kernel_s_axis_tvalid && kernel_s_axis_tready && kernel_s_axis_tlast)) begin // reset buffer
            buffer_cnt <= 0;
        end
        else if(kernel_s_axis_tvalid && kernel_s_axis_tready && buffer_cnt<KERNEL_SIZE)begin // fill next
            kernel_buffer[buffer_cnt] <= kernel_s_axis_tdata[31:0];
            buffer_cnt <= buffer_cnt +1;
        end
    end
    
    // out valid//
    always @(posedge kernel_mclk)begin
        if (kernel_m_axis_tvalid)begin
            if(kernel_m_axis_tready) begin // ready next
                kernel_m_axis_tvalid <= (kernel_m_axis_tlast && buffer_cnt != KERNEL_SIZE) ? 0 : 1;
                kernel_m_axis_tlast <= out_cnt == KERNEL_SIZE - 2 ? 1 : 0;
                kernel_m_axis_tdata <= out_cnt == KERNEL_SIZE - 1 ? kernel_buffer[0] : kernel_buffer[out_cnt + 1];
                kernel_m_axis_tkeep <= 4'b1111;
                out_cnt <= out_cnt == KERNEL_SIZE - 1 ? 0 : out_cnt + 1;
            end
        end
        else begin // initialization
            kernel_m_axis_tvalid <= buffer_cnt == KERNEL_SIZE ? 1 : 0;
            kernel_m_axis_tlast <= 0;
            kernel_m_axis_tdata <= kernel_buffer[0];
            kernel_m_axis_tkeep <= 4'b1111;
            out_cnt <= 0;
        end
    end
{% endhighlight %}

For the image data pipeline, we also use DMA input and output. The idea is same but it comes with more complication to deal with the delays for the extra clocks needed for BRAM read/write, DSP slice processing, and accumulation, which is too long to write down here unfortunately.

Let's write a testbench to test the operation by sending some AXI stream data.
{% highlight verilog %}
    integer i,y,x; 
    initial begin
        #(1); // hold time
        kernel_s_tvalid_reg = 0;
        kernel_m_tready_reg = 0;
        conv_s_tvalid_reg = 0;
        conv_m_tready_reg = 0;
        #(10*PER);
        kernel_s_tvalid_reg = 1;
        for(i=0;i<25;i=i+1) begin
            #(PER)kernel_s_tdata_reg = i;
            /*
            0  ~  4
            ~ ... ~
            20 ~ 24
            */
            kernel_s_tlast_reg = i==24 ? 1 : 0;
        end
        #(PER)kernel_s_tvalid_reg = 0;
        kernel_s_tlast_reg = 0;
        #(10*PER);
        for(i=0;i<25;i=i+1) begin
            #(PER)kernel_m_tready_reg = 1;
            #(PER)kernel_m_tready_reg = 0;
        end
        
        #(10*PER);
        for(y=0;y<5;y=y+1) begin
            conv_s_tvalid_reg = 1;
            for(x=0;x<640;x=x+1)begin
                conv_s_tdata_reg = 100*y + x;
                conv_s_tlast_reg = x==639 ? 1 : 0;
                #PER;
                while(!conv_s_tready)#PER;
            end
            conv_s_tvalid_reg = 0;
            for(x=0;x<640;x=x+1)begin
                conv_s_tlast_reg = x==639 ? 1 : 0;
                while(!conv_m_tvalid)#PER;
                conv_m_tready_reg = 1;
                #PER conv_m_tready_reg = 0;
                #PER;
            end
        end
    end
{% endhighlight %}
<br>![Simulation Result](/assets/images/pynq_conv/sim_res.PNG)<br>

Now we connect our modules in the block diagram. We use two DMA blocks: one for kernel data and the other for image/result data.
<br>![Block Diagram](/assets/images/pynq_conv/block_diagram.PNG)<br>

And we synthesize/implement the design for the PYNQ board we'll use.
<br>![Implementation](/assets/images/pynq_conv/implementation.PNG)<br>

The IPs can be accessed with python scripts through Jupyter notebook.
{% highlight python %}
from pynq.lib import AxiGPIO
from pynq import Overlay, allocate
import numpy as np
ol = Overlay("design_1.bit")

def assign_kernel_buffer(input_buffer):
    ol.kernel_data_dma.sendchannel.transfer(input_buffer)
    ol.kernel_data_dma.sendchannel.wait()
def read_kernel_buffer(output_buffer):
    ol.kernel_data_dma.recvchannel.transfer(output_buffer)
    ol.kernel_data_dma.recvchannel.wait()

image = np.random.randint(0,1024,(480,640)).astype('u4')
kernel = np.arange(25).reshape(5,5).astype('u4')

with allocate(shape=(25,), dtype=np.uint32) as input_buffer:    
    with allocate(shape=(25,), dtype=np.uint32) as output_buffer:        
        # reset
        input_buffer[-1] = 0xffff_ffff
        assign_kernel_buffer(input_buffer)
        
        input_buffer[:] = np.arange(25)
        assign_kernel_buffer(input_buffer)
        read_kernel_buffer(output_buffer) #flush
        read_kernel_buffer(output_buffer)

fpga_result = np.empty((480,640),dtype='u4')
with allocate(shape=(480,640,), dtype=np.uint32) as input_buffer:    
    with allocate(shape=(480,640,), dtype=np.uint32) as output_buffer:
        input_buffer[:] = image
        for input_row, output_row in zip(input_buffer,output_buffer):
            ol.convolution_data_dma.sendchannel.transfer(input_row)
            ol.convolution_data_dma.recvchannel.transfer(output_row)
            ol.convolution_data_dma.sendchannel.wait()
            ol.convolution_data_dma.recvchannel.wait()
        fpga_result[:] = output_buffer
{% endhighlight %}

And we can validate the result with the scipy library.

{% highlight python %}
from scipy.ndimage import convolve
convolved = convolve(image,kernel)
assert(not (fpga_result[4:-4,4:-4] - convolved[2:-6,2:-6]).any())
{% endhighlight %}
<br>![result](/assets/images/pynq_conv/jupyter_result.PNG)<br>