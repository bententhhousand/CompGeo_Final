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
#include <CGAL/draw_surface_mesh.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Search_traits_d.h>
#include <CGAL/Octree.h>
#include <CGAL/Bbox_3.h>

#include <cstdlib>
#include <vector>
#include <fstream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel 	Kernel;
typedef Kernel::FT                                          	FT;
typedef Kernel::Point_3                                     	Point_3;
typedef Kernel::Vector_3                                    	Vector_3;
typedef Kernel::Sphere_3                                    	Sphere_3;
typedef CGAL::Point_set_3<Point_3, Vector_3>                	Point_set;
typedef Point_set::Point_map                                    Point_map;
typedef CGAL::Search_traits_3<Kernel>                          	Traits;
typedef CGAL::Kd_tree<Traits> 									kTree;
typedef std::vector<Point_3> 									Point_vector;
typedef CGAL::Octree<Kernel, Point_set, Point_map>              Octree;
typedef CGAL::Bbox_3                                            Bbox;
typedef CGAL::Surface_mesh<Point_3>                             Mesh;
typedef Mesh::Vertex_index                                      Vertex_descriptor;
typedef CGAL::Orthtrees::Preorder_traversal                     Preorder_traversal;

void printBbox(Bbox& inputBox){
    
    std::cout << "X-Min:" << inputBox.xmin() << std::endl;
    std::cout << "X-Max:" << inputBox.xmax() << std::endl;
    std::cout << "Y-Min:" << inputBox.ymin() << std::endl;
    std::cout << "Y-Max:" << inputBox.ymax() << std::endl;
    std::cout << "Z-Min:" << inputBox.zmin() << std::endl;
    std::cout << "Z-Max:" << inputBox.zmax() << std::endl;

}

void add_square_to_mesh(Mesh &mesh, double xmin, double xmax, double ymin, double ymax) {
    Point_3 p1(xmin, ymin, 0);
    Point_3 p2(xmax, ymin, 0);
    Point_3 p3(xmax, ymax, 0);
    Point_3 p4(xmin, ymax, 0);

    Vertex_descriptor v1 = mesh.add_vertex(p1);
    Vertex_descriptor v2 = mesh.add_vertex(p2);
    Vertex_descriptor v3 = mesh.add_vertex(p3);
    Vertex_descriptor v4 = mesh.add_vertex(p4);

    mesh.add_edge(v1, v2);
    mesh.add_edge(v2, v3);
    mesh.add_edge(v3, v4);
    mesh.add_edge(v4, v1);
}

void add_cube_to_mesh(Mesh &mesh, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax) {
    Point_3 p1(xmin, ymin, zmin);
    Point_3 p2(xmax, ymin, zmin);
    Point_3 p3(xmin, ymax, zmin);
    Point_3 p4(xmax, ymax, zmin);
    Point_3 p5(xmin, ymin, zmax);
    Point_3 p6(xmax, ymin, zmax);
    Point_3 p7(xmin, ymax, zmax);
    Point_3 p8(xmax, ymax, zmax);

    Vertex_descriptor v1 = mesh.add_vertex(p1);
    Vertex_descriptor v2 = mesh.add_vertex(p2);
    Vertex_descriptor v3 = mesh.add_vertex(p3);
    Vertex_descriptor v4 = mesh.add_vertex(p4);
    Vertex_descriptor v5 = mesh.add_vertex(p5);
    Vertex_descriptor v6 = mesh.add_vertex(p6);
    Vertex_descriptor v7 = mesh.add_vertex(p7);
    Vertex_descriptor v8 = mesh.add_vertex(p8);

    mesh.add_edge(v1, v2);
    mesh.add_edge(v2, v4);
    mesh.add_edge(v4, v3);
    mesh.add_edge(v3, v1);

    mesh.add_edge(v5, v6);
    mesh.add_edge(v6, v8);
    mesh.add_edge(v8, v7);
    mesh.add_edge(v7, v5);

    mesh.add_edge(v1, v5);
    mesh.add_edge(v2, v6);
    mesh.add_edge(v3, v7);
    mesh.add_edge(v4, v8);

}

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


	kTree ktree;
	Point_vector pv;
    Mesh inputMesh;

	for (Point_3 p : points.points()){
        inputMesh.add_vertex(p);
	}

	CGAL::draw(inputMesh);

    Octree octree(points, points.point_map());
    octree.refine(100000, 1);
    typedef std::array<std::size_t, 3> Facet;
    std::vector<Facet> facets;
    CGAL::advancing_front_surface_reconstruction(points.points().begin(),
                                                 points.points().end(),
                                                 std::back_inserter(facets));
    std::cout << facets.size ()
              << " facet(s) generated by reconstruction." << std::endl;
    // copy points for random access
    std::vector<Point_3> vertices;
    vertices.reserve (points.size());
    std::copy (points.points().begin(), points.points().end(), std::back_inserter (vertices));
    Mesh output_mesh;
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh (vertices, facets, output_mesh);
    CGAL::draw(output_mesh);
    
    for (Octree::Node node : octree.traverse<Preorder_traversal>()) {
        Bbox nodeBox = octree.bbox(node);
        //add_square_to_mesh(inputMesh, nodeBox.xmin(), nodeBox.xmax(), nodeBox.ymin(), nodeBox.ymax());
        add_cube_to_mesh(inputMesh, nodeBox.xmin(), nodeBox.xmax(), nodeBox.ymin(), nodeBox.ymax(), nodeBox.zmin(), nodeBox.zmax());
        //std::cout << node << "\n" << std::endl;
    }
    CGAL::draw(inputMesh);
}
