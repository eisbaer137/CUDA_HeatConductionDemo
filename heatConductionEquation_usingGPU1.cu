// this program simulates heat dissipation from square-shaped rods with constant temperatures
// the governing equation is heat conduction equation laplacian T = k * dT/dt,
// where k is a given heat conduction constant and T=T(x,y;t) is a temperature at a point (x,y) and time t
// simulation is done in two-dimensional space.
// cuda texture memory is used for calculating T
// written by H. H. Yoo

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "misc.h"
#include "cpu_anim.h"

#define SCR_WIDTH   1024
#define PI  3.1415926535897932f
#define MAX_TEMP    1.0f
#define MIN_TEMP 0.0001f
#define TIMELUMP    5

const float alpha = 25.0f;
const float dt = 0.01f;
const float ds = 1.0f;

const float k = alpha * dt / (ds * ds);

texture<float, 2> texConstSrc;
texture<float, 2> texIn;
texture<float, 2> texOut;

__global__ void temporalUpdateCells(float* src, bool turn)
{   
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = x + y * blockDim.x * gridDim.x;

    float top, left, right, bottom, center;
    if(turn)
    {
        top = tex2D(texIn, x, y - 1);
        left = tex2D(texIn, x - 1, y);
        right = tex2D(texIn, x + 1, y);
        bottom = tex2D(texIn, x, y + 1);
        center = tex2D(texIn, x, y);
    }
    else
    {
        top = tex2D(texOut, x, y - 1);
        left = tex2D(texOut, x - 1, y);
        right = tex2D(texOut, x + 1, y);
        bottom = tex2D(texOut, x, y + 1);
        center = tex2D(texOut, x, y + 1);
    }
    src[tid] = center + k * (top + left + right + bottom 
                -4 * center);
}

__global__ void keepConstCells(float* src)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = x + y * blockDim.x * gridDim.x;
    
    float center = tex2D(texConstSrc, x, y);
    if(center != 0.0f)
        src[tid] = center;

}

class DataBlock
{
public:
    unsigned char* output_bitmap;
    float* dev_inSrc;
    float* dev_outSrc;
    float* dev_constSrc;
    CPUAnimBitmap* bitmap;
};

void anim_gpu(DataBlock* db, int ticks)
{
    dim3 block(16, 16);
    dim3 grid((SCR_WIDTH + block.x - 1) / block.x,
            (SCR_WIDTH + block.y - 1) / block.y);
        
    CPUAnimBitmap* bitmap = db->bitmap;
    float* in, *out;
    bool turn = true;
    for(int i=0; i<TIMELUMP; i++)
    {
        if(turn)
        {
            in = db->dev_inSrc;
            out = db->dev_outSrc;
        }
        else
        {
            in = db->dev_outSrc;
            out = db->dev_inSrc;
        }
        keepConstCells<<<grid, block>>>(in);
        temporalUpdateCells<<<grid, block>>>(out, turn);
        turn = !turn;
    }
    float_to_color<<<grid, block>>>(db->output_bitmap, db->dev_inSrc);
    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), db->output_bitmap,
                bitmap->image_size(), cudaMemcpyDeviceToHost));
}

void anim_exit(DataBlock* db)
{
    cudaUnbindTexture(texIn);
    cudaUnbindTexture(texOut);
    cudaUnbindTexture(texConstSrc);
    HANDLE_ERROR(cudaFree(db->dev_inSrc));
    HANDLE_ERROR(cudaFree(db->dev_outSrc));
    HANDLE_ERROR(cudaFree(db->dev_constSrc));
}


int main(int argc, char** argv)
{   
    DataBlock db;
    CPUAnimBitmap bitmap(SCR_WIDTH, SCR_WIDTH, &db);
    float* heatSource;
    int hx, hy;

    db.bitmap = &bitmap;
    int imgSize = bitmap.image_size();

    HANDLE_ERROR(cudaMalloc((void**)&db.output_bitmap, imgSize));
    HANDLE_ERROR(cudaMalloc((void**)&db.dev_inSrc, imgSize));
    HANDLE_ERROR(cudaMalloc((void**)&db.dev_outSrc, imgSize));
    HANDLE_ERROR(cudaMalloc((void**)&db.dev_constSrc, imgSize));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    HANDLE_ERROR(cudaBindTexture2D(NULL, texConstSrc, db.dev_constSrc,
                desc, SCR_WIDTH, SCR_WIDTH, SCR_WIDTH * sizeof(float)));
    HANDLE_ERROR(cudaBindTexture2D(NULL, texIn, db.dev_inSrc,
                desc, SCR_WIDTH, SCR_WIDTH, SCR_WIDTH * sizeof(float)));
    HANDLE_ERROR(cudaBindTexture2D(NULL, texOut, db.dev_outSrc,
                desc, SCR_WIDTH, SCR_WIDTH, SCR_WIDTH * sizeof(float)));

    heatSource = (float*)malloc(imgSize);
    for(int i=0; i<SCR_WIDTH * SCR_WIDTH; i++)
    {
        heatSource[i] = 0.0f;
        hx = i % SCR_WIDTH;
        hy = i / SCR_WIDTH;
        // setting up heat rods that are kept in constant temperature at MAX_TEMP
        if(hx>200 && hx<250 && hy>700 && hy<750)
            heatSource[i] = MAX_TEMP;
        if(hx>400 && hx<450 && hy>700 && hy<750)
            heatSource[i] = MAX_TEMP;
        if(hx>600 && hx<650 && hy>700 && hy<750)
            heatSource[i] = MAX_TEMP;

        if(hx>200 && hx<250 && hy>550 && hy<600)
            heatSource[i] = MAX_TEMP;
        if(hx>400 && hx<450 && hy>550 && hy<600)
            heatSource[i] = MAX_TEMP;
        if(hx>600 && hx<650 && hy>550 && hy<600)
            heatSource[i] = MAX_TEMP;

        if(hx>200 && hx<250 && hy>400 && hy<450)
            heatSource[i] = MAX_TEMP;
        if(hx>400 && hx<450 && hy>400 && hy<450)
            heatSource[i] = MAX_TEMP;
        if(hx>600 && hx<650 && hy>400 && hy<450)
            heatSource[i] = MAX_TEMP;
    
    
    }

    HANDLE_ERROR(cudaMemcpy(db.dev_constSrc, heatSource, imgSize,
                        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(db.dev_inSrc, heatSource, imgSize,
                        cudaMemcpyHostToDevice));

    free(heatSource);

    bitmap.anim_and_exit((void (*)(void*, int))anim_gpu,
                            (void (*)(void*))anim_exit);


    return 0;
}


