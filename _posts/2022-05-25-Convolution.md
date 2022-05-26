---
title:  "[ISP]Image Convolution With GPU / OpenCL"
date:   2022-05-25 00:00:00 +0900
categories: jekyll update
layout: post
---

Convolution filter not only sees widespread practical use over multiple domains, but also serves as a great practice tool for hardware acceleration.
For simplicity's sake, we will impose some constraints.
1. Inputs are two images of same width and height.
2. The width and height is powers of two. (In usual, it can be made by padding zeros)
3. Convolution is wrapped. (Again, zeros are padded normally)
4. We might calculate cross correlation instead of convolution. The two are not the same, however the differentiation is just a matter of semantics in terms of algorithms.

First we start with the definition of convolution as a brute force algorithm.
{% highlight python %}
def convolve(image1, image2):
    h, w = image1.shape
    padded = np.tile(image1,(2,2)) # Wrapping the input
    ret = np.empty_like(image1)
    for y,row in enumerate(ret):
        for x,c in enumerate(row):
            ret[y,x] = (padded[y:y+h,x:x+h]*image2[::-1,::-1]).sum()
    return ret
{% endhighlight %}

There are so many problems with this approach, but it suffices to say the time complexity is *O(w1×h1×w2×h2)*. It's actually manageable when and only when the filter size is very small.
The preferable approach for larger kernel size is using Fourier transform.
1. Convolution is multiplication over fourier domain.
2. Multiplication has time complexity of only *O(w×h)*.

Discrete Fourier transform of a 2D image can be implemented by applying fourier transform over each axes.

{% highlight python %}
def DFT(image): # Descrete Fourier Transform over the first axis
    assert image.dtype == np.complex64
    h, w = image.shape
    ret = np.empty_like(image, dtype = 'complex64') # int32_t real, int32_t imag
    yi, xi = np.indices((h, h), dtype = 'complex64')
    matrix = np.exp(yi*xi*(np.pi * -2j / h))
    return matrix@image

def DFT_2D(image): # DFT over both axes, consequentially
    return DFT(DFT(image).T).T
{% endhighlight %}

The time complexity for DFT is *O(w<sup>2</sup>×h+w×h<sup>2</sup>)*. This is quite an improvement already.

{% highlight python %}
def flip_scale(image):
    image[1:,:] = image[-1:0:-1,:]
    image[:,1:] = image[:,-1:0:-1]
    image /= image.size

def convolve_DFT(image1, image2):
    Fourier = DFT_2D(image1) * DFT_2D(image2)
    ret = DFT_2D(Fourier)
    flip_scale(ret)
    return ret
{% endhighlight %}

Inverse DFT is basically DFT with a little bit of flipping and scaling.

{% highlight python %}
from functools import lru_cache
def FFT(image): # Cooley - Tukey Divde and Conquer. Top-Down Approach.
    assert image.dtype == np.complex64
    out = image.copy()
    @lru_cache
    def get_vector(h):
        return np.exp(np.indices((h//2,), dtype = 'complex64') * (np.pi * -2j / h)).reshape(-1,1)
    def FFT_half_sample(out):
        h, w = out.shape
        if h == 1:return
        assert h%2 == 0
        FFT_half_sample(out[::2]) # EVEN
        FFT_half_sample(out[1::2]) # ODD
        vector = get_vector(h)
        out[:h//2], out[h//2:] = out[::2] + vector * out[1::2], out[::2] - vector * out[1::2]
    FFT_half_sample(out)
    return out

def FFT_2D(image):
    return FFT(FFT(image).T).T

def convolve_FFT(image1, image2):
    Fourier = FFT_2D(image1) * FFT_2D(image2)
    ret = FFT_2D(Fourier)
    flip_scale(ret)
    return ret
{% endhighlight %}

DFT can be made faster with Fast Fourier Transform(FFT), as in its name. In short DFT can be calculated from DFT with half sampling rate, recursively.
The time complexity is reduced down to *O(wlog(w)×h+w×hlog(h))*.

It's still heavy calculation. Let's improve it even further with the help of GPGPU hardware accleration.
We will use non-proprietary OpenCL for this task. We can also use CUDA instead and the kernel languages are not so different from each other except for some jargons.

{% highlight C %}
#define WP(X,Y) (X<Y?X:X-Y) //WRAP
#define CC_BS 32 // Block size
__kernel void cross_correlate(__global const float *A, __global const float *B, __global double *C){ // Block Size 32 x 32
  const int y = get_global_id(0);
  const int x = get_global_id(1);
  const int h = get_global_size(0);
  const int w = get_global_size(1);
  const int ty = get_local_id(0);
  const int tx = get_local_id(1);
  __local float shared_A[CC_BS][CC_BS];
  __local float shared_B[CC_BS*2][CC_BS*2];
  double ret = 0;
  for(int sy = 0; sy < h; sy += CC_BS)for(int sx = 0; sx < w; sx += CC_BS){
    shared_A[ty][tx] = A[WP(sy+ty,h) * w + WP(sx+tx,w)];
    shared_B[ty][tx] = B[WP(sy+y,h) * w + WP(sx+x,w)];
    shared_B[ty+CC_BS][tx] = B[WP(sy+y+CC_BS,h) * w + WP(sx+x,w)];
    shared_B[ty][tx+CC_BS] = B[WP(sy+y,h) * w + WP(sx+x+CC_BS,w)];
    shared_B[ty+CC_BS][tx+CC_BS] = B[WP(sy+y+CC_BS,h) * w + WP(sx+x+CC_BS,w)];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int dy=0;dy<CC_BS;dy++)for(int dx=0;dx<CC_BS;dx++){
      ret += (double)(shared_A[dy][dx] * shared_B[dy + ty][dx + tx]);
    }
  }
  C[y*w + x] = ret;
}
{% endhighlight %}

There are many things to be considered to make faster kernels.
1. Minimize global memory acces and maximize shared memory access.
2. Consider the limit to shared memory size.
3. There is also a limit to number of threads per block.
4. GPU memory is much more expensive and smaller.
5. All of these will also vary by hardware specification.
6. Cache hits might or might not help. When it works it works great, but relies too much on heuristics.
7. Maximize bandwidth. Let the data flow.
There is no one true general answer to this and it takes a lot of tirals and errors to optimize.

{% highlight python %}
def GPU_conv(A,B):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C = np.empty_like(A)
    res_g = cl.Buffer(ctx, mf.READ_WRITE, C.nbytes)
    prg.cross_correlate(queue, A.shape, (32,32), a_g, b_g, res_g)
    cl.enqueue_copy(queue, C, res_g)
    return C
{% endhighlight %}

We'll use pyopencl for kernel calls. No matter how parallelized this is with all the GPU cores, this is still *O(w1×h1×w2×h2)*.
Let's make DFT and FFT kernels too.
A small trick here is using different kernels depending on the input size. 
There is no kernel that fits all because of GPU memory constraints.

{% highlight C %}
#define TPB_Y {local_size[1]}

inline float2 comp_mul(float2 a, float2 b){
  return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
// exp(iw) = cosw + sinw*i
inline float2 comp_exp(float2 a){
  return (float2)(exp(a.x)) * (float2)(cos(a.y), sin(a.y));
}
inline float2 exp_i(float a){
  return (float2)(cos(a), sin(a));
}
__kernel void DFT(__global const float2 *A, __global float2 *B){ // 1D over 1D
  const int n = get_global_size(1);
  const int By = get_global_id(1);
  const int ty = get_local_id(1);
  const int ly = get_global_size(0);
  const int row = get_global_id(0);
  __private float2 sum = (0, 0);
  __local float2 shared[TPB_Y];
  // without shared memory : (n global reads) + (1 global writes) per thread
  // with shared memory : (n/TPB global reads) + (n/TPB global writes) per thread
  for(int Ay = 0; Ay < n; Ay += TPB_Y){
    if(Ay+ty<n)shared[ty] = A[n*row + Ay + ty];
    barrier(CLK_LOCAL_MEM_FENCE); // load shared memory and synchronize
    for(int x = Ay; x < Ay + TPB_Y; x++)sum += comp_mul(exp_i(-2*M_PI_F/n*x*By), shared[x-Ay]);
  }
  B[n*row + By] = sum;
}
__kernel void FFT_rearrange(__global const float2 *A, __global float2 *B){ // 1D(X) over 2D(Y,X)
  const int nx = get_global_size(1);
  const int ty = get_global_id(0);
  const int tx = get_global_id(1);
  private int n = nx;
  private int dx = 0, x=tx;
  while(n>1){
    dx <<= 1;
    dx += x%2;
    x>>=1;
    n>>=1;
  }
  B[nx*ty + dx] = A[nx*ty + tx];
}
__kernel void FFT_under_512(__global float2 *A){ // kernel size 1x512, in place
  const int nx = get_global_size(1);
  const int y = get_global_id(0);
  const int x = get_global_id(1);
  const int tx = get_local_id(1);
  __local float2 shared[512]; // 9 recursion, 1 global read, 1 global write
  shared[tx] = A[nx*y + x]; // global read
  barrier(CLK_LOCAL_MEM_FENCE);
  int odd, dx, hx;
  float2 temp;
  for(int i=1;i<=9;i++){
    dx = (tx>>i)<<i;
    odd = tx & ( 1 << (i-1) );
    hx = (tx^dx) & (odd-1);
    temp = shared[dx + hx] + (odd?-1:1) * comp_mul(exp_i(-2*M_PI_F/(1 << i)*hx), shared[dx + (1 << (i-1)) + hx] );
    barrier(CLK_LOCAL_MEM_FENCE);
    shared[tx] = temp;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  A[nx*y + x] = shared[tx]; // global write
}
__kernel void FFT_over_512(__global const float2 *A, __global float2 *B , const int i){
  const int nx = get_global_size(1);
  const int y = get_global_id(0);
  const int x = get_global_id(1);
  int odd, dx, hx;
  dx = (x>>i)<<i;
  odd = x & ( 1 << (i-1) );
  hx = (x^dx) & (odd-1);
  B[nx * y + x] = A[dx + hx] + (odd?-1:1) * comp_mul(exp_i(-2*M_PI_F/(1 << i)*hx), A[dx + (1 << (i-1)) + hx] );
}
{% endhighlight %}
And the kernels are invoked with pyopencl.
{% highlight python %}
def GPU_FFT(A):
    a_np = A.astype('c8')
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    res_g = cl.Buffer(ctx, mf.READ_WRITE, a_np.nbytes)
    tmp = cl.Buffer(ctx, mf.READ_WRITE, a_np.nbytes)
    # knl = prg.FFT_rearrange
    prg.FFT_rearrange(queue, a_np.shape, None, a_g, res_g)
    prg.FFT_under_512(queue, a_np.shape, (1,512), res_g)
    i = np.array(10,dtype='i4')
    while (1<<i) <= A.shape[0]:
        res_g, tmp = tmp, res_g
        prg.FFT_over_512(queue, a_np.shape, None, tmp, res_g, i)
        i += 1
    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np

def GPU_DFT(A):
    a_np = A.astype('c8')
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    local_size = (1,1024)
    prg.DFT(queue, a_np.shape, local_size, a_g, res_g)
    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np

def GPU_FFT_2D(A):
    return GPU_FFT(GPU_FFT(A).T).T
def GPU_DFT_2D(A):
    return GPU_DFT(GPU_DFT(A).T).T

def GPU_conv_FFT(A,B):
    return GPU_FFT_2D(GPU_FFT_2D(A)*GPU_FFT_2D(B))
def GPU_conv_DFT(A,B):
    return GPU_DFT_2D(GPU_DFT_2D(A)*GPU_DFT_2D(B))
{% endhighlight %}

The actual computing time is measured as below.
<br>![Time measure](/assets/images/Convolution/result.png)<br>
We achieved quite an acceleration without heavy optimization.