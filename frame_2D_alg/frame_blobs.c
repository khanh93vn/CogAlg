#include <stdio.h>
#include <malloc.h>
#include "bitarray.h"

typedef struct {
    int x, y;
} DertRef;

typedef struct {
    double I, G, Dy, Dx;
    unsigned long long S;
    char sign, fopen;
    DertRef *dert_ref;
} Blob;

typedef struct {
    Blob *blobs_ptr;
    int nblobs;
} FrameOfBlobs;

int adj_offset[8][2] = {
    {-1, 0},    /* top */
    {0, 1},     /* right */
    {1, 0},     /* bottom */
    {0, -1},    /* left */
    {-1, -1},   /* top-left */
    {-1, 1},    /* top-right */
    {1, 1},     /* btm-right */
    {1, -1},    /* btm-left */
};

FrameOfBlobs derts2blobs(double *i_, double *g_, double *dy_, double *dx_,
                          int height, int width, int ave,
                          unsigned int *idmap) {

    Blob    *blobs;
    DertRef *dert_refs;
    int nfilled = 0,                    /* filled derts count */
        nblobs = 0,                     /* total number of blobs */
        *queue, qlen, qbeg, qend,       /* a queue for FIFO data */
        *fill_map,                      /* id_map to track flood fill */
        size = height * width;          /* total number of derts */

    // Memory allocation
    blobs = (Blob*) malloc((size/2 + 2)*sizeof(Blob));  /* worst case scenario number of blobs */
    dert_refs = (DertRef*) malloc(size * sizeof(DertRef));
    fill_map = (int*) malloc((size/32 + 1) * 4);
    qlen = height<width?width:height + 1;
    queue = (int*) malloc(qlen * sizeof(int));

    clearbits(fill_map, size);


    // Loop through all derts
    for(int i = 0; i < size; i++)
        if(!testbit(fill_map, i)) {  /* ignore filled derts */
            setbit(fill_map, i);    /* set current dert as filled */
            blobs[nblobs].dert_ref = &dert_refs[nfilled];  /* save pointer, length is S */
            double I = 0, G = 0, Dy = 0, Dx = 0, S = 0;
            char sign = g_[i] - ave > 0,
                 fopen = 0;

            // do flood fill
            qbeg = 0;
            qend = 1;
            queue[qbeg] = i;
            while(qbeg != qend) {
                int j = queue[qbeg++];           /* pop last dert's index */
                if(qbeg >= qlen) qbeg = 0;
                I += i_[j];
                G += g_[j];
                Dy += dy_[j];
                Dx += dx_[j];
                S++;
                idmap[j] = nblobs;

                int y = j / width,
                    x = j % width;
                dert_refs[nfilled].x = x;  /* save filled dert position */
                dert_refs[nfilled].y = y;
                nfilled++;

                // loop through adjacent coordinates, 8 if sign else 4
                for(int dir = 0; dir < (sign?8:4); dir++) {
                    int y2 = y + adj_offset[dir][0],
                        x2 = x + adj_offset[dir][1];
                    // check if image boundary is reached
                    if(y2 < 0 || y2 >= height ||
                       x2 < 0 || x2 >= width) fopen = 1;
                    else {
                        int k = y2 * width + x2;
                        // ignore filled
                        if(testbit(fill_map, k)) continue;
                        // check if same-signed
                        if(sign == (g_[k] - ave > 0)) {
                            setbit(fill_map, k);    /* set current dert as filled */
                            queue[qend++] = k;  /* append this hash */
                            if(qend >= qlen) qend = 0;
                        }
                        // else assign adjacents
                        else {
                            // TODO: assign adjacents
                        }
                    }
                }
            }
            blobs[nblobs].I = I;
            blobs[nblobs].G = G;
            blobs[nblobs].Dy = Dy;
            blobs[nblobs].Dx = Dx;
            blobs[nblobs].S = S;
            blobs[nblobs].sign = sign;
            blobs[nblobs].fopen = fopen;
            nblobs++;
        }
    // free memory
    free(fill_map);
    free(queue);

    // return frame of blobs
    realloc(blobs, nblobs * sizeof(Blob));
    FrameOfBlobs *frame;
    frame = (FrameOfBlobs*) malloc(sizeof(FrameOfBlobs));
    frame->blobs_ptr = blobs;
    frame->nblobs = nblobs;

    return *frame;
}
