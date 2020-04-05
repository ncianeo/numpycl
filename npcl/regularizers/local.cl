__kernel void grad(
    __global const float *input,
    __global float * output
    ){
    
    int i = get_global_id(0);
    int j = get_global_id(1);

    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    output[i+Nx*j] = i<Nx-1?input[i+1+Nx*j]-input[i+Nx*j]:0;
    output[Nx*Ny+i+Nx*j] = j<Ny-1?input[i+Nx*(j+1)]-input[i+Nx*j]:0;
}


__kernel void norm(
    __global const float * input,
    __global float *output
    ){

    int i = get_global_id(0);
    int j = get_global_id(1);

    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    output[i+Nx*j] = sqrt(input[i+Nx*j]*input[i+Nx*j] + input[Nx*Ny+i+Nx*j]*input[Nx*Ny+i+Nx*j]);
}


__kernel void divide_3d_by_2d(
    __global float *p,
    __global const float *norm
    ){

    int i = get_global_id(0);
    int j = get_global_id(1);

    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    p[i+Nx*j] /= norm[i+Nx*j];
    p[Nx*Ny+i+Nx*j] /= norm[i+Nx*j];
}


__kernel void divergence(
    __global const float * input,
    __global float * output
    ){

    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float px, py;
    px = i>0?input[i+Nx*j]-input[i-1+Nx*j]:0;
    py = j>0?input[Nx*Ny+i+Nx*j]-input[Nx*Ny+i+Nx*(j-1)]:0;

    output[i+Nx*j] = px+py;
}