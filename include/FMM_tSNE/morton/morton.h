#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <glm/glm.hpp>
#include <vector>

namespace Morton
{

inline uint32_t spread16(uint32_t x)
{
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x;
}

inline uint32_t compact16(uint32_t x)
{
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    x = (x | (x >> 8)) & 0x0000FFFFu;
    return x;
}

inline void createMortonCode(std::vector<TsnePoint2D>& points, glm::dvec2 negMinPos, double largestAxis)
{
    glm::dvec2 position{0.0};
    for (int i = 0; i < points.size(); i++)
    {
        position = points[i].position;

        position = (position + negMinPos) / (largestAxis); // rescale such that range is between [0-1]

        position = glm::min(
            glm::max(position * 65536.0, glm::dvec2(0.0)),
            glm::dvec2(65535.0)); // rescale such that range is between [0-65535] which is the max number for 16 bits
#ifdef INDEX_TRACKER
        points[i].morton_code = (spread16((uint32_t) position.y) << 1) | spread16((uint32_t) position.x);
#endif
    }
}

} // namespace Morton
