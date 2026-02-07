#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Kinetic_surface_reconstruction_3.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/bounding_box.h>
#include <iostream>
#include <string>

using Kernel    = CGAL::Exact_predicates_inexact_constructions_kernel;
using FT        = typename Kernel::FT;
using Point_3   = typename Kernel::Point_3;
using Vector_3  = typename Kernel::Vector_3;

using Point_set    = CGAL::Point_set_3<Point_3>;
using Point_map    = typename Point_set::Point_map;
using Normal_map   = typename Point_set::Vector_map;

using KSR = CGAL::Kinetic_surface_reconstruction_3<Kernel, Point_set, Point_map, Normal_map>;

void print_usage(const char* program_name) {
    std::cout << "\n=== CGAL Kinetic Surface Reconstruction ===" << std::endl;
    std::cout << "\nUsage: " << program_name << " <input.ply> [options]\n" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -lambda <value>     Complexity vs fidelity: 0.3-0.99 (default: 0.5)" << std::endl;
    std::cout << "                      Lower = more detail, Higher = simpler mesh" << std::endl;
    std::cout << "  -dist <value>       Max distance to plane (default: auto)" << std::endl;
    std::cout << "                      Smaller = detects smaller features" << std::endl;
    std::cout << "  -angle <value>      Max normal angle deviation in degrees (default: 10)" << std::endl;
    std::cout << "  -minpts <value>     Min points per region (default: 50)" << std::endl;
    std::cout << "                      Smaller = detects smaller planar regions" << std::endl;
    std::cout << "  -k <value>          Partition complexity: 1-4 (default: 2)" << std::endl;
    std::cout << "  -output <filename>  Output filename (default: reconstructed.ply)" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " building.ply -lambda 0.3 -minpts 20" << std::endl;
    std::cout << "  " << program_name << " building.ply -dist 0.01 -lambda 0.5" << std::endl;
    std::cout << "  " << program_name << " building.ply -lambda 0.3 -minpts 10 -dist 0.05" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    
    // Default parameters
    std::string input_file = argv[1];
    std::string output_file = "reconstructed.ply";
    FT lambda = 0.5;
    FT max_distance = -1;  // -1 means auto-calculate
    FT max_angle = 10;
    int min_region_size = 50;
    int k_intersections = 2;
    
    // Parse command line arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-lambda" && i + 1 < argc) {
            lambda = std::stod(argv[++i]);
        } else if (arg == "-dist" && i + 1 < argc) {
            max_distance = std::stod(argv[++i]);
        } else if (arg == "-angle" && i + 1 < argc) {
            max_angle = std::stod(argv[++i]);
        } else if (arg == "-minpts" && i + 1 < argc) {
            min_region_size = std::stoi(argv[++i]);
        } else if (arg == "-k" && i + 1 < argc) {
            k_intersections = std::stoi(argv[++i]);
        } else if (arg == "-output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }
    
    // Load point cloud
    Point_set point_set;
    if (!CGAL::IO::read_point_set(input_file, point_set)) {
        std::cerr << "Error: Cannot read " << input_file << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "\n=== CGAL Kinetic Surface Reconstruction ===" << std::endl;
    std::cout << "\nInput: " << input_file << std::endl;
    std::cout << "Points loaded: " << point_set.size() << std::endl;
    
    // Calculate bounding box for scale info
    CGAL::Bbox_3 bbox = CGAL::bbox_3(
        CGAL::make_transform_iterator_from_property_map(point_set.begin(), point_set.point_map()),
        CGAL::make_transform_iterator_from_property_map(point_set.end(), point_set.point_map()));
    
    FT diagonal = std::sqrt(
        (bbox.xmax() - bbox.xmin()) * (bbox.xmax() - bbox.xmin()) +
        (bbox.ymax() - bbox.ymin()) * (bbox.ymax() - bbox.ymin()) +
        (bbox.zmax() - bbox.zmin()) * (bbox.zmax() - bbox.zmin()));
    
    std::cout << "\nBounding box:" << std::endl;
    std::cout << "  X: [" << bbox.xmin() << ", " << bbox.xmax() << "]" << std::endl;
    std::cout << "  Y: [" << bbox.ymin() << ", " << bbox.ymax() << "]" << std::endl;
    std::cout << "  Z: [" << bbox.zmin() << ", " << bbox.zmax() << "]" << std::endl;
    std::cout << "  Diagonal: " << diagonal << std::endl;
    
    // Auto-calculate max_distance if not provided
    if (max_distance < 0) {
        max_distance = diagonal * 0.01;  // 1% of diagonal
        std::cout << "\nAuto-calculated max_distance: " << max_distance << std::endl;
    }
    
    // Estimate normals if not present
    if (!point_set.has_normal_map()) {
        std::cout << "\nEstimating normals..." << std::endl;
        point_set.add_normal_map();
        CGAL::pca_estimate_normals<CGAL::Parallel_if_available_tag>(point_set, 12);
        CGAL::mst_orient_normals(point_set, 12);
        std::cout << "Normals estimated and oriented." << std::endl;
    } else {
        std::cout << "\nNormals already present in point cloud." << std::endl;
    }
    
    // Print parameters being used
    std::cout << "\n--- Parameters ---" << std::endl;
    std::cout << "  lambda (complexity): " << lambda << std::endl;
    std::cout << "  max_distance: " << max_distance << std::endl;
    std::cout << "  max_angle: " << max_angle << " degrees" << std::endl;
    std::cout << "  min_region_size: " << min_region_size << " points" << std::endl;
    std::cout << "  k_intersections: " << k_intersections << std::endl;
    
    // Set parameters
    auto param = CGAL::parameters::maximum_distance(max_distance)
        .maximum_angle(max_angle)
        .minimum_region_size(min_region_size)
        .k_neighbors(12)
        .reorient_bbox(true)
        .regularize_parallelism(true)
        .regularize_coplanarity(true)
        .angle_tolerance(5)
        .maximum_offset(max_distance * 0.2);
    
    // Initialize and run
    std::cout << "\n--- Processing ---" << std::endl;
    std::cout << "Running shape detection..." << std::endl;
    KSR ksr(point_set, param);
    ksr.detection_and_partition(k_intersections, param);
    
    std::cout << "Detected " << ksr.detected_planar_shapes().size() 
              << " planar shapes" << std::endl;
    
    // Reconstruct
    std::cout << "Reconstructing surface..." << std::endl;
    
    std::vector<Point_3> vertices;
    std::vector<std::vector<std::size_t>> faces;
    
    ksr.reconstruct_with_ground(lambda, 
        std::back_inserter(vertices), 
        std::back_inserter(faces));
    
    if (faces.size() > 0) {
        CGAL::IO::write_polygon_soup(output_file, vertices, faces);
        std::cout << "\n=== SUCCESS ===" << std::endl;
        std::cout << "Output: " << output_file << std::endl;
        std::cout << "Vertices: " << vertices.size() << std::endl;
        std::cout << "Faces: " << faces.size() << std::endl;
        
        // Suggestions for tuning
        std::cout << "\n--- Tips ---" << std::endl;
        std::cout << "Too simple? Try: -lambda 0.3 -minpts " << min_region_size/2 << std::endl;
        std::cout << "Too complex? Try: -lambda 0.8 -minpts " << min_region_size*2 << std::endl;
        
        return EXIT_SUCCESS;
    } else {
        std::cerr << "\nReconstruction failed - no faces generated" << std::endl;
        std::cerr << "Try: -lambda 0.3 -minpts 20 -dist " << max_distance * 2 << std::endl;
        return EXIT_FAILURE;
    }
}
