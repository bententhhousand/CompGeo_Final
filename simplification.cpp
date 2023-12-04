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
#include <CGAL/Kd_tree_node.h>
#include <CGAL/Kd_tree_rectangle.h>
#include <CGAL/Search_traits_d.h>
#include <CGAL/Octree.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/point_generators_3.h>

#include <cstdlib>
#include <vector>
#include <fstream>
#include <math.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel 	Kernel;
typedef Kernel::FT                                          	FT;
typedef Kernel::Point_3                                     	Point_3;
typedef Kernel::Vector_3                                    	Vector_3;
typedef Kernel::Sphere_3                                    	Sphere_3;
typedef CGAL::Point_set_3<Point_3, Vector_3>                	Point_set;
typedef Point_set::Point_map                                    Point_map;
typedef CGAL::Search_traits_3<Kernel>                          	Traits;
typedef CGAL::Kd_tree<Traits> 									Kd_Tree;
typedef Kd_Tree::Node                                           Node;
typedef Kd_Tree::Internal_node                                  Internal_node;
typedef Kd_Tree::Leaf_node                                      Leaf_node;
typedef Kd_Tree::Node_handle                                    Node_handle;
typedef CGAL::Fair<Traits>                                      Fair;
typedef CGAL::Kd_tree_rectangle<FT, CGAL::Dimension_tag<3>>     Kd_tree_rectangle;
typedef std::vector<Point_3> 									Point_vector;
typedef CGAL::Octree<Kernel, Point_set, Point_map>              Octree;
typedef CGAL::Bbox_3                                            Bbox;
typedef CGAL::Surface_mesh<Point_3>                             Mesh;
typedef Mesh::Vertex_index                                      Vertex_descriptor;
typedef CGAL::Orthtrees::Preorder_traversal                     Preorder_traversal;
typedef CGAL::Orthtrees::Leaves_traversal                     	Leaves_traversal;

void print_kd_bounds(const Kd_tree_rectangle& rect) {
    // Get the bounds for each dimension
    FT xmin = rect.min_coord(0);
    FT xmax = rect.max_coord(0);
    FT ymin = rect.min_coord(1);
    FT ymax = rect.max_coord(1);
    FT zmin = rect.min_coord(2);
    FT zmax = rect.max_coord(2);

    // Print the bounds
    std::cout << "X Bounds: [" << xmin << ", " << xmax << "]\n";
    std::cout << "Y Bounds: [" << ymin << ", " << ymax << "]\n";
    std::cout << "Z Bounds: [" << zmin << ", " << zmax << "]\n";
}

double euclidean_distance(Point_3 a, Point_3 b) {
    return sqrt(pow(a.x() - b.x(), 2) + pow(a.y() - b.y(), 2) + pow(a.z() - b.z(), 2));
}

double nearest_point_distance(Point_set &set1, Point_set &set2) {
    
    double final_distance = euclidean_distance(set1.point(*set1.begin()), set2.point(*set2.begin()));
    for (Point_set::const_iterator it = set1.begin(); it != set1.end(); it++) {
        double min_distance = euclidean_distance(set1.point(*it), set2.point(*set2.begin()));
        for (Point_set::const_iterator jt = set2.begin(); jt != set2.end(); jt++) {
            double distance = euclidean_distance(set1.point(*it), set2.point(*jt));
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
        if (min_distance < final_distance){
			final_distance = min_distance;
		}
    }
    return final_distance;
}

double average_point_distance(Point_set &set1, Point_set &set2) {
    
    double final_distance = 0;
    for (Point_set::const_iterator it = set1.begin(); it != set1.end(); it++) {
        for (Point_set::const_iterator jt = set2.begin(); jt != set2.end(); jt++) {
            final_distance += euclidean_distance(set1.point(*it), set2.point(*jt));
        }
    }
    return final_distance / (set1.size() * set2.size());
}

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
Point_3 bbox_center(Bbox bbox){
	double x_center = (bbox.xmin() + bbox.xmax())/2.0;
	double y_center = (bbox.ymin() + bbox.ymax())/2.0;
	double z_center = (bbox.zmin() + bbox.zmax())/2.0;

	return Point_3(x_center, y_center, z_center);
}

Point_3 bbox_center_kd(const Kd_tree_rectangle& bbox){
    double x_center = (bbox.min_coord(0) + bbox.max_coord(0))/2.0;
	double y_center = (bbox.min_coord(1) + bbox.max_coord(1))/2.0;
	double z_center = (bbox.min_coord(2) + bbox.max_coord(2))/2.0;

	return Point_3(x_center, y_center, z_center);
}

void construct_kd_mesh(const Node* node, const Kd_tree_rectangle& bbox, Mesh &mesh, Point_set &averagedPoints) {
    if (!node) return;

    if (node -> is_leaf()) {
        // Process the points in the leaf node
        const Leaf_node* leaf_node = static_cast<const Leaf_node*>(node);
        averagedPoints.insert(bbox_center_kd(bbox));
    } else {
        const Internal_node* internal_node = static_cast<const Internal_node*>(node);
        // print_kd_bounds(bbox);
        add_cube_to_mesh(mesh, bbox.min_coord(0), bbox.max_coord(0), bbox.min_coord(1), bbox.max_coord(1), bbox.min_coord(2), bbox.max_coord(2));
        // Split the bounding box
        Kd_tree_rectangle lower_bbox(bbox);
        Kd_tree_rectangle upper_bbox(bbox);
        internal_node->split_bbox(lower_bbox, upper_bbox);

        // Now you can use lower_bbox and upper_bbox for further processing

        // Recursively traverse the child nodes
        construct_kd_mesh(internal_node->lower(), lower_bbox, mesh, averagedPoints);
        construct_kd_mesh(internal_node->upper(), upper_bbox, mesh, averagedPoints);
    }
}


Point_set octree_middles(Octree octree){

	Point_set set;

	double counts = 0;
	int number = 0;

	for (Octree::Node node : octree.traverse<Leaves_traversal>()) {
		if (!node.empty()){
			counts += node.size();
        	set.insert(bbox_center(octree.bbox(node)));
			number ++;
		}
    }

	std::cout << "Average = " << counts/number << std::endl;

	return set;
}


// Point_set octree_averages(Octree octree){
// 	Point_set set;

// 	int countSum = 0;
// 	int countCount = 0;

// 	for (Octree::Node node : octree.traverse<Leaves_traversal>()) {
// 		if (!node.empty()){
// 			double xAvg = 0;
// 			double zAvg = 0;
// 			double yAvg = 0;
// 			int count = 0;
// 			auto currentPoint = node.begin();
// 			while (currentPoint != node.end()){
// 				Point_3 point = octree.point(*currentPoint);
// 				xAvg += point.x();
// 				yAvg += point.y();
// 				zAvg += point.z();
// 				count ++;
// 			}
//         	set.insert(Point_3(xAvg/count, yAvg/count, zAvg/count));
// 			countSum += count;
// 			countCount ++;
// 		}
//     }

// 	std::cout << "Average of " << countSum / countCount << " points per filled box" << endl;

// 	return set;
// }

int main(int argc, char* argv[]){ 
	// cmake -G"Visual Studio 16" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/Users/matth/Documents/CompGeo/vcpkg/scripts/buildsystems/vcpkg.cmake ..
    // cmake --build .

    Point_set points;
    if (argc < 3)
    {
        std::cerr << "Usage: [input.xyz], [bucket size]" << std::endl;
    }
    std::string fileName = argv[1];
    int bucketSize = std::stoi(argv[2]);

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


	Point_vector pv;
    Mesh inputMesh;
    Mesh inputKdMesh;
	for (Point_3 p : points.points()){
        inputMesh.add_vertex(p);
        inputKdMesh.add_vertex(p);
	}

	Kd_Tree ktree(points.points().begin(), points.points().end(), bucketSize);
	//CGAL::draw(inputMesh);

    Octree octree(points, points.point_map());
    octree.refine(100000, bucketSize);
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
    //CGAL::draw(output_mesh);
    
    for (Octree::Node node : octree.traverse<Preorder_traversal>()) {
        Bbox nodeBox = octree.bbox(node);
        //add_square_to_mesh(inputMesh, nodeBox.xmin(), nodeBox.xmax(), nodeBox.ymin(), nodeBox.ymax());
        add_cube_to_mesh(inputMesh, nodeBox.xmin(), nodeBox.xmax(), nodeBox.ymin(), nodeBox.ymax(), nodeBox.zmin(), nodeBox.zmax());
        //std::cout << node << "\n" << std::endl;
    }
    //CGAL::draw(inputMesh);

    // Test point distance
    Point_set cloud1;
    Point_set cloud2;

    cloud1.insert(Point_3(1, 0, 0));
    cloud2.insert(Point_3(1, 0, 5));

	std::cout << "Point cloud distances (Should both be 5): " << nearest_point_distance(cloud1, cloud2) << ", " <<nearest_point_distance(cloud2, cloud1) << std::endl;

    cloud2.insert(Point_3(1, 0, 10));

	std::cout << "Point cloud distances (Should still both be 5): " << nearest_point_distance(cloud1, cloud2) << ", " <<nearest_point_distance(cloud2, cloud1) << std::endl;

    cloud2.insert(Point_3(1, 0, 2));

	std::cout << "Point cloud distances (Should both be 2): " << nearest_point_distance(cloud1, cloud2) << ", " <<nearest_point_distance(cloud2, cloud1) << std::endl;

    cloud2.insert(Point_3(1, 0, 0));

	std::cout << "Point cloud distances (Should both be 0): " << nearest_point_distance(cloud1, cloud2) << ", " <<nearest_point_distance(cloud2, cloud1) << std::endl;

    // cloud1.insert(Point_3(4, 5, 6));
    // cloud1.insert(Point_3(7, 8, 9));

    // cloud2.insert(Point_3(4, 5, 6));
    // cloud2.insert(Point_3(7, 8, 10));
    // cloud2.insert(Point_3(1, 2.5, 3));
    // cloud2.insert(Point_3(7, 1, 10));

    // double distance = nearest_point_distance(cloud1, cloud2);
    // std::cout << "Point cloud Distance: " << distance << std::endl;
	// std::cout << "Other distance: "<< nearest_point_distance(cloud2, cloud1) << std::endl;
    // Iterate over kd tree
    Point_set kdMiddles;
    construct_kd_mesh(ktree.root(), ktree.bounding_box(), inputKdMesh, kdMiddles);
    //CGAL::draw(inputKdMesh);

	CGAL::draw(points);
	Point_set middles = octree_middles(octree);

    
    std::cout << "KD-Tree Simplification" << std::endl;
    
	CGAL::draw(kdMiddles);
	std::cout << nearest_point_distance(points, kdMiddles) << ", " << average_point_distance(points, kdMiddles) << std::endl;

    std::cout << "Octree Simplification" << std::endl;
    
	CGAL::draw(middles);
	std::cout << nearest_point_distance(points, middles) << ", " << average_point_distance(points, middles) << std::endl;


}