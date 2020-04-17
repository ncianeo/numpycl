__kernel void grad(
    __global const float *input,
    __global float * gx,
    __global float * gy
    ){
    
    int i = get_global_id(0);
    int j = get_global_id(1);

    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    gx[i+Nx*j] = i<Nx-1?input[i+1+Nx*j]-input[i+Nx*j]:0;
    gy[i+Nx*j] = j<Ny-1?input[i+Nx*(j+1)]-input[i+Nx*j]:0;
}


__kernel void norm(
    __global const float * gx,
    __global const float * gy,
    __global float *output
    ){

    int i = get_global_id(0);
    int j = get_global_id(1);

    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    output[i+Nx*j] = sqrt(pow(gx[i+Nx*j], 2) + pow(gy[i+Nx*j], 2));
}


__kernel void divergence2d(
    __global const float * px,
    __global const float * py,
    __global float * output
    ){

    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float dx, dy;
    dx = i>0&&i<Nx-1?px[i+Nx*j]-px[i-1+Nx*j]:i==0?px[i+Nx*j]:-px[i-1+Nx*j];
    dy = j>0&&j<Ny-1?py[i+Nx*j]-py[i+Nx*(j-1)]:j==0?py[i+Nx*j]:-py[i+Nx*(j-1)];

    output[i+Nx*j] = dx+dy;
}
