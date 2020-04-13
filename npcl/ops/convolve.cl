inline wrap(int x, int w){
    int res = x % w;
    return res<0?res+w:res;
}

__kernel void transpose2d(
    __global const float * input,
    __global float * output
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    output[i + Nx*j] = input[(Nx-1-i) + Nx*(Ny-1-j)];
}


__kernel void transpose2d_sv(
    __global const float * input,
    __global float * output,
    const int Nhx,
    const int Nhy
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    uint ref_i, ref_j;
    #pragma unroll
    for(int dy = -Nhy/2; dy< Nhy/2+1; ++dy){
        ref_j = clamp(j+dy, 0, Ny-1);
        #pragma unroll
        for(int dx = -Nhx/2; dx< Nhx/2+1; ++dx){
            ref_i = clamp(i+dx, 0, Nx-1);
            output[Nx*Ny*(Nhx/2+dx+Nhx*(Nhy/2+dy))+i+Nx*j] = input[Nx*Ny*(Nhx/2-dx+Nhx*(Nhy/2-dy))+ref_i+Nx*ref_j];
        }
    }
}


__kernel void convolve2d_w(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhx,
    const int Nhy
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    #pragma unroll
    for(int m = -Nhx/2; m < Nhx/2+1; ++m){
        ref_i = wrap(i+m, Nx);
        #pragma unroll
        for(int n = -Nhy/2; n< Nhy/2+1; ++n){
            ref_j = wrap(j+n, Ny);
            res += h[Nhx/2+m + Nhx*(Nhy/2+n)]*input[ref_i+Nx*ref_j];
        }
    }
    output[i+Nx*j] = res;
}


__kernel void convolve2d_s(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhx,
    const int Nhy
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    #pragma unroll
    for(int hty = 0; hty< Nhy; ++hty){
        ref_j = clamp(j-Nhy/2+hty, 0, Ny-1);
        #pragma unroll
        for(int htx = 0; htx< Nhx; ++htx){
            ref_i = clamp(i-Nhx/2+htx, 0, Nx-1);
            res += h[htx+hty*Nhx]*input[ref_i+ref_j*Nx];
        }
    }
    output[i+Nx*j] = res;
}


__kernel void convolve2d_z(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhx,
    const int Nhy
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    const int x_start = (i-Nhx/2)<0?Nhx/2-i:0;
    const int x_end = (i+Nhx/2)>Nx-1?Nhx/2-i+Nx-1:Nhx;
    const int y_start = (j-Nhy/2)<0?Nhy/2-j:0;
    const int y_end = (j+Nhy/2)>Ny-1?Nhy/2-j+Ny-1:Nhy;


    #pragma unroll
    for(int hty=y_start; hty<y_end; ++hty){
        ref_j = j-Nhy/2+hty;
        #pragma unroll
        for(int htx=x_start; htx<x_end; ++htx){
            ref_i = i-Nhx/2+htx;
            res += h[htx+hty*Nhx]*input[ref_i+ref_j*Nx];
        }
    }
    output[i+Nx*j] = res;
}


__kernel void convolve2d_sv_w(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhx,
    const int Nhy
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    #pragma unroll
    for(int hty=0; hty<Nhy; ++hty){
        ref_j = (j-Nhy/2+hty) % Ny;
        #pragma unroll
        for(int htx=0; htx<Nhx; ++htx){
            ref_i = (i-Nhx/2+htx) % Nx;
            res += h[Nx*Ny*(htx+hty*Nhx)+i+Nx*j]*input[ref_i+ref_j*Nx];
        }
    }
    output[i+Nx*j] = res;
}


__kernel void convolve2d_sv_s(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhx,
    const int Nhy
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    #pragma unroll
    for(int hty=0; hty<Nhy; ++hty){
        ref_j = clamp(j-Nhy/2+hty, 0, Ny-1);
        #pragma unroll
        for(int htx=0; htx<Nhx; ++htx){
            ref_i = clamp(i-Nhx/2+htx, 0, Nx-1);
            res += h[Nx*Ny*(htx+hty*Nhx)+i+Nx*j]*input[ref_i+ref_j*Nx];
        }
    }
    output[i+Nx*j] = res;
}


__kernel void convolve2d_sv_z(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhx,
    const int Nhy
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    const int x_start = (i-Nhx/2)<0?Nhx/2-i:0;
    const int x_end = (i+Nhx/2)>Nx-1?Nhx/2-i+Nx-1:Nhx;
    const int y_start = (j-Nhy/2)<0?Nhy/2-j:0;
    const int y_end = (j+Nhy/2)>Ny-1?Nhy/2-j+Ny-1:Nhy;

    #pragma unroll
    for(int hty=y_start; hty<y_end; ++hty){
        ref_j = j-Nhy/2+hty;
        #pragma unroll
        for(int htx=x_start; htx<x_end; ++htx){
            ref_i = i-Nhx/2+htx;
            res += h[Nx*Ny*(htx+hty*Nhx)+i+Nx*j]*input[ref_i+ref_j*Nx];
        }
    }
    output[i+Nx*j] = res;
}