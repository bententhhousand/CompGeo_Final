#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/remove_outliers.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/jet_smooth_point_set.h>
#include <CGAL/jet_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Scale_space_surface_reconstruction_3.h>
#include <CGAL/Scale_space_reconstruction_3/Jet_smoother.h>
#include <CGAL/Scale_space_reconstruction_3/Advancing_front_mesher.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/draw_point_set_3.h>

#include <cstdlib>
#include <vector>
#include <fstream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT                                          FT;
typedef Kernel::Point_3                                     Point_3;
typedef Kernel::Vector_3                                    Vector_3;
typedef Kernel::Sphere_3                                    Sphere_3;
typedef CGAL::Point_set_3<Point_3, Vector_3>                Point_set;

int main(int argc, char* argv[]){ // cmake -G"Visual Studio 16" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/Users/matth/Documents/CompGeo/vcpkg/scripts/buildsystems/vcpkg.cmake ..
    // cmake --build .

    Point_set points;
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " [input.xyz/off/ply/las]" << std::endl;
        std::cerr <<"Running " << argv[0] << " data/kitten.xyz -1\n";
    }
    std::string fileName = argv[1];

    std::ifstream stream (fileName, std::ios_base::binary);
    if (!stream)
    {
        std::cerr << "Error: cannot read file " << fileName << std::endl;
        return EXIT_FAILURE;
    }
    stream >> points;
    std::cout << "Read " << points.size () << " point(s)" << std::endl;
    if (points.empty())
        return EXIT_FAILURE;

    CGAL::draw(points);

}