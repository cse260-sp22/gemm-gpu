
#include "mytypes.h"
#include <stdio.h>

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{

   // set your block dimensions and grid dimensions here
   if(n < 300) {
      gridDim.x = n / (BLOCKTILE_N * 2);
      gridDim.y = n / (BLOCKTILE_M * 2);
   }
   else {
      gridDim.x = n / (BLOCKTILE_N * 8);
      gridDim.y = n / (BLOCKTILE_M * 8);
   }
   // you can overwrite blockDim here if you like.
   if (n % BLOCKTILE_N != 0)
      gridDim.x++;
   if (n % BLOCKTILE_M != 0)
      gridDim.y++;
}
