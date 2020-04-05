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


__kernel void convolve2d(
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


__kernel void convolve2d_sv(
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
            res += h[Nx*Ny*(htx+hty*Nhx)+i+Nx*j]*input[ref_i+ref_j*Nx];
        }
    }
    output[i+Nx*j] = res;
}