inline int wrap(int x, int w){
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
    const int Nhy,
    const int Nhx
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    uint ref_i, ref_j;
    #pragma unroll
    for(int dx = -Nhx/2; dx <= Nhx/2; ++dx){
        ref_i = clamp(i+dx, 0, Nx-1);
        #pragma unroll
        for(int dy = -Nhy/2; dy <= Nhy/2; ++dy){
            ref_j = clamp(j+dy, 0, Ny-1);    
            output[Nx*Ny*(Nhx/2+dx+Nhx*(Nhy/2+dy))+i+Nx*j] = input[Nx*Ny*(Nhx/2-dx+Nhx*(Nhy/2-dy))+ref_i+Nx*ref_j];
        }
    }
}


__kernel void convolve2d_w(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhy,
    const int Nhx
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    #pragma unroll
    for(int dx = -Nhx/2; dx <= Nhx/2; ++dx){
        ref_i = wrap(i+dx, Nx);
        #pragma unroll
        for(int dy = -Nhy/2; dy <= Nhy/2; ++dy){
            ref_j = wrap(j+dy, Ny);
            res += h[Nhx/2+dx + Nhx*(Nhy/2+dy)]*input[ref_i+Nx*ref_j];
        }
    }
    output[i+Nx*j] = res;
}


__kernel void convolve2d_s(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhy,
    const int Nhx
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    #pragma unroll
    for(int dx = -Nhx/2; dx <= Nhx/2; ++dx){
        ref_i = clamp(i+dx, 0, Nx-1);
        #pragma unroll
        for(int dy = -Nhy/2; dy <= Nhy/2; ++dy){
            ref_j = clamp(j+dy, 0, Ny-1);
            res += h[Nhx/2+dx + Nhx*(Nhy/2+dy)]*input[ref_i+Nx*ref_j];
        }
    }
    output[i+Nx*j] = res;
}


__kernel void convolve2d_z(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhy,
    const int Nhx
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    #pragma unroll
    for(int dx = max(-Nhx/2, -i); dx <= min(Nhx/2, Nx-1-i); ++dx){
        ref_i = i+dx;
        #pragma unroll
        for(int dy = max(-Nhy/2, -j); dy <= min(Nhy/2, Ny-1-j); ++dy){
            ref_j = j+dy;
            res += h[Nhx/2+dx + Nhx*(Nhy/2+dy)]*input[ref_i+Nx*ref_j];
        }
    }
    output[i+Nx*j] = res;
}

// Using Local Memories

__kernel void convolve2d_loc_z(
    __global const float *input,
    __global const float *h,
    __local float *P,
    __global float *output,
    const int Ny,
    const int Nx,
    const int FSY,
    const int FSX
    ){
    int i_g = get_global_id(0);
    int j_g = get_global_id(1);
    
    int i_loc = get_local_id(0);
    int j_loc = get_local_id(1);
    
    int HFSX = FSX / 2;
    int HFSY = FSY / 2;
    int PSX = 2 * HFSX;
    int PSY = 2 * HFSY;
    int TS = get_local_size(0);
    
    if (i_g < Nx && j_g < Ny){
        P[i_loc + HFSX + (TS+PSX)*(j_loc + HFSY)] = input[i_g + Nx * j_g];
        
        // This padding is valid only if FSX < TS and FSY < TS
        int xoffset = 0;
        int yoffset = 0;
        if (i_loc < HFSX){
            xoffset = -min(i_g, HFSX);
        }
        if (i_loc >= TS-HFSX){
            xoffset = min(HFSX, Nx-1-i_g);
        }
        if (j_loc < HFSY){
            yoffset = -min(j_g, HFSY);
        }
        if (j_loc >= TS-HFSY){
            yoffset = min(HFSY, Ny-1-j_g);
        }
        if (xoffset!=0){
            P[i_loc + HFSX + xoffset + (TS+PSX)*(j_loc + HFSY)] = input[i_g + xoffset + Nx * j_g];
        }
        if (yoffset!=0){
            P[i_loc + HFSX + (TS+PSX)*(j_loc + HFSY + yoffset)] = input[i_g + Nx * (j_g + yoffset)];
        }
        if (xoffset!=0 && yoffset!=0){
            P[i_loc + HFSX + xoffset + (TS+PSX)*(j_loc + HFSY + yoffset)] = input[i_g + xoffset + Nx * (j_g + yoffset)];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float sum = 0.0;
        int i_ref, j_ref;
        #pragma unroll
        for (int dx = max(-HFSX, -i_g); dx <= min(HFSX, Nx-1-i_g); dx++){
            i_ref = i_loc + HFSX + dx;
            #pragma unroll
            for (int dy = max(-HFSY, -j_g); dy <= min(HFSY, Ny-1-j_g); dy++){
                j_ref = j_loc + HFSY + dy;
                sum += P[i_ref + (TS+PSX)*j_ref] * h[dx + HFSX + FSX * (dy + HFSY)];
            }
        }
        output[i_g + Nx * j_g] = sum;
    }
    else{
        return;
    }
}


__kernel void convolve2d_loc_s(
    __global const float *input,
    __global const float *h,
    __local float *P,
    __global float *output,
    const int Ny,
    const int Nx,
    const int FSY,
    const int FSX
    ){
    int i_g = get_global_id(0);
    int j_g = get_global_id(1);
    
    int i_loc = get_local_id(0);
    int j_loc = get_local_id(1);
    
    int HFSX = FSX / 2;
    int HFSY = FSY / 2;
    int PSX = 2 * HFSX;
    int PSY = 2 * HFSY;
    int TS = get_local_size(0);
    
    if (i_g < Nx && j_g < Ny){
        P[i_loc + HFSX + (TS+PSX)*(j_loc + HFSY)] = input[i_g + Nx * j_g];
        
        // This padding is valid only if FSX < TS and FSY < TS
        int xoffset = 0;
        int yoffset = 0;
        if (i_loc < HFSX){
            xoffset = -HFSX;
        }
        if (i_loc >= TS-HFSX){
            xoffset = HFSX;
        }
        if (j_loc < HFSY){
            yoffset = -HFSY;
        }
        if (j_loc >= TS-HFSY){
            yoffset = HFSY;
        }
        if (xoffset!=0){
            P[i_loc + HFSX + xoffset + (TS+PSX)*(j_loc + HFSY)] =
                input[clamp(i_g + xoffset, 0, Nx-1) + xoffset + Nx * j_g];
        }
        if (yoffset!=0){
            P[i_loc + HFSX + (TS+PSX)*(j_loc + HFSY + yoffset)] = 
                input[i_g + Nx * clamp(j_g + yoffset, 0, Ny-1)];
        }
        if (xoffset!=0 && yoffset!=0){
            P[i_loc + HFSX + xoffset + (TS+PSX)*(j_loc + HFSY + yoffset)] = 
                input[clamp(i_g + xoffset, 0, Nx-1) + Nx * clamp(j_g + yoffset, 0, Ny-1)];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float sum = 0.0;
        int i_ref, j_ref;
        #pragma unroll
        for (int dx = -HFSX; dx <= HFSX; dx++){
            i_ref = i_loc + HFSX + dx;
            #pragma unroll
            for (int dy = -HFSY; dy <= HFSY; dy++){
                j_ref = j_loc + HFSY + dy;
                sum += P[i_ref + (TS+PSX)*j_ref] * h[dx + HFSX + FSX * (dy + HFSY)];
            }
        }
        output[i_g + Nx * j_g] = sum;
    }
    else{
        return;
    }
}


__kernel void convolve2d_loc_w(
    __global const float *input,
    __global const float *h,
    __local float *P,
    __global float *output,
    const int Ny,
    const int Nx,
    const int FSY,
    const int FSX
    ){
    int i_g = get_global_id(0);
    int j_g = get_global_id(1);
    
    int i_loc = get_local_id(0);
    int j_loc = get_local_id(1);
    
    int HFSX = FSX / 2;
    int HFSY = FSY / 2;
    int PSX = 2 * HFSX;
    int PSY = 2 * HFSY;
    int TS = get_local_size(0);
    
    if (i_g < Nx && j_g < Ny){
        P[i_loc + HFSX + (TS+PSX)*(j_loc + HFSY)] = input[i_g + Nx * j_g];
        
        // This padding is valid only if FSX < TS and FSY < TS
        int xoffset = 0;
        int yoffset = 0;
        if (i_loc < HFSX){
            xoffset = -HFSX;
        }
        if (i_loc >= TS-HFSX){
            xoffset = HFSX;
        }
        if (j_loc < HFSY){
            yoffset = -HFSY;
        }
        if (j_loc >= TS-HFSY){
            yoffset = HFSY;
        }
        if (xoffset!=0){
            P[i_loc + HFSX + xoffset + (TS+PSX)*(j_loc + HFSY)] =
                input[wrap(i_g + xoffset, Nx) + xoffset + Nx * j_g];
        }
        if (yoffset!=0){
            P[i_loc + HFSX + (TS+PSX)*(j_loc + HFSY + yoffset)] =
                input[i_g + Nx * wrap(j_g + yoffset, Ny)];
        }
        if (xoffset!=0 && yoffset!=0){
            P[i_loc + HFSX + xoffset + (TS+PSX)*(j_loc + HFSY + yoffset)] =
                input[wrap(i_g + xoffset, Nx) + Nx * wrap(j_g + yoffset, Ny)];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float sum = 0.0;
        int i_ref, j_ref;
        #pragma unroll
        for (int dx = -HFSX; dx <= HFSX; dx++){
            i_ref = i_loc + HFSX + dx;
            #pragma unroll
            for (int dy = -HFSY; dy <= HFSY; dy++){
                j_ref = j_loc + HFSY + dy;
                sum += P[i_ref + (TS+PSX)*j_ref] * h[dx + HFSX + FSX * (dy + HFSY)];
            }
        }
        output[i_g + Nx * j_g] = sum;
    }
    else{
        return;
    }
}


    __kernel void convolve2d_sv_w(
        __global const float *input,
        __global const float *h,
        __global float *output,
        const int Nhy,
        const int Nhx
        ){
        int i = get_global_id(0);
        int j = get_global_id(1);
        int Nx = get_global_size(0);
        int Ny = get_global_size(1);

        float res = 0.f;
        uint ref_i, ref_j;

        #pragma unroll
        for(int dx = -Nhx/2; dx <= Nhx/2; ++dx){
            ref_i = wrap(i+dx, Nx);
            #pragma unroll
            for(int dy = -Nhy/2; dy <= Nhy/2; ++dy){
                ref_j = wrap(j+dy, Ny);
                res += h[Nx*Ny*(Nhx/2+dx + Nhx*(Nhy/2 + dy))+i+Nx*j]*input[ref_i + Nx * ref_j];
            }
        }
        output[i+Nx*j] = res;
}


__kernel void convolve2d_sv_s(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhy,
    const int Nhx
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    #pragma unroll
    for(int dx = -Nhx/2; dx <= Nhx/2; ++dx){
        ref_i = clamp(i+dx, 0, Nx-1);
        #pragma unroll
        for(int dy = -Nhy/2; dy <= Nhy/2; ++dy){
            ref_j = clamp(j+dy, 0, Ny-1);
            res += h[Nx*Ny*(Nhx/2+dx + Nhx*(Nhy/2 + dy))+i+Nx*j]*input[ref_i + Nx * ref_j];
        }
    }
    output[i+Nx*j] = res;
}


__kernel void convolve2d_sv_z(
    __global const float *input,
    __global const float *h,
    __global float *output,
    const int Nhy,
    const int Nhx
    ){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    uint ref_i, ref_j;

    #pragma unroll
    for(int dx = max(-Nhx/2, -i); dx <= min(Nhx/2, Nx-1-i); ++dx){
        ref_i = i+dx;
        #pragma unroll
        for(int dy = max(-Nhy/2, -j); dy <= min(Nhy/2, Ny-1-j); ++dy){
            ref_j = j+dy;
            res += h[Nx*Ny*(Nhx/2+dx + Nhx*(Nhy/2 + dy))+i+Nx*j]*input[ref_i + Nx * ref_j];
        }
    }
    output[i+Nx*j] = res;
}