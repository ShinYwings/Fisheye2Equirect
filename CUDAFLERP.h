

#include <cstdint>

class bilin_inter
{
    public:
    void CUDAFLERP(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, unsigned char* __restrict const d_out, const uint32_t neww, const uint32_t newh);
}
