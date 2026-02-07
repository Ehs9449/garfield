#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using FT = typename Kernel::FT;
using Point_3 = typename Kernel::Point_3;
using Vector_3 = typename Kernel::Vector_3;

using Point_set = CGAL::Point_set_3<Point_3>;
using Point_map = typename Point_set::Point_map;
using Normal_map = typename Point_set::Vector_map;

// RANSAC typedefs
using Traits = CGAL::Shape_detection::Efficient_RANSAC_traits<Kernel, Point_set, Point_map, Normal_map>;
using Efficient_ransac = CGAL::Shape_detection::Efficient_RANSAC<Traits>;
using Plane = CGAL::Shape_detection::Plane<Traits>;
using Cylinder = CGAL::Shape_detection::Cylinder<Traits>;
using Cone = CGAL::Shape_detection::Cone<Traits>;
using Sphere = CGAL::Shape_detection::Sphere<Traits>;
using Torus = CGAL::Shape_detection::Torus<Traits>;

void print_usage(const char* program_name) {
    std::cout << "\n=== CGAL RANSAC Shape Detection ===" << std::endl;
    std::cout << "\nUsage: " << program_name << " <input.ply> [options]\n" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -probability <value>   Probability to miss smallest shape (default: 0.05)" << std::endl;
    std::cout << "  -min_points <value>    Min points per shape (default: 100)" << std::endl;
    std::cout << "  -epsilon <value>       Max distance to primitive (default: auto)" << std::endl;
    std::cout << "  -cluster <value>       Max cluster epsilon (default: auto)" << std::endl;
    std::cout << "  -normal <value>        Max normal deviation in degrees (default: 25)" << std::endl;
    std::cout << "  -all                   Detect all primitives (planes, cylinders, etc.)" << std::endl;
    std::cout << "  -output <filename>     Output filename (default: ransac_detected.ply)" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " building.ply -min_points 50" << std::endl;
    std::cout << "  " << program_name << " building.ply -epsilon 0.01 -min_points 20" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Check for help flag
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }
    
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    
    // Default parameters
    std::string input_file = argv[1];
    std::string output_file = "ransac_detected.ply";
    FT probability = 0.05;
    int min_points = 100;
    FT epsilon = -1;
    FT cluster_epsilon = -1;
    FT normal_threshold = 25;
    bool detect_all = false;
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-probability" && i + 1 < argc) {
            probability = std::stod(argv[++i]);
        } else if (arg == "-min_points" && i + 1 < argc) {
            min_points = std::stoi(argv[++i]);
        } else if (arg == "-epsilon" && i + 1 < argc) {
            epsilon = std::stod(argv[++i]);
        } else if (arg == "-cluster" && i + 1 < argc) {
            cluster_epsilon = std::stod(argv[++i]);
        } else if (arg == "-normal" && i + 1 < argc) {
            normal_threshold = std::stod(argv[++i]);
        } else if (arg == "-all") {
            detect_all = true;
        } else if (arg == "-output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    // Load point cloud
    Point_set point_set;
    if (!CGAL::IO::read_point_set(input_file, point_set)) {
        std::cerr << "Error: Cannot read " << input_file << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "\n=== CGAL RANSAC Shape Detection ===" << std::endl;
    std::cout << "\nInput: " << input_file << std::endl;
    std::cout << "Points loaded: " << point_set.size() << std::endl;
    
    // Calculate bounding box
    CGAL::Bbox_3 bbox = CGAL::bbox_3(
        CGAL::make_transform_iterator_from_property_map(point_set.begin(), point_set.point_map()),
        CGAL::make_transform_iterator_from_property_map(point_set.end(), point_set.point_map()));
    
    FT diagonal = std::sqrt(
        (bbox.xmax() - bbox.xmin()) * (bbox.xmax() - bbox.xmin()) +
        (bbox.ymax() - bbox.ymin()) * (bbox.ymax() - bbox.ymin()) +
        (bbox.zmax() - bbox.zmin()) * (bbox.zmax() - bbox.zmin()));
    
    std::cout << "Bounding box diagonal: " << diagonal << std::endl;
    
    // Auto-calculate epsilon
    if (epsilon < 0) epsilon = diagonal * 0.01;
    if (cluster_epsilon < 0) cluster_epsilon = diagonal * 0.01;
    
    // Estimate normals if not present
    if (!point_set.has_normal_map()) {
        std::cout << "\nEstimating normals..." << std::endl;
        point_set.add_normal_map();
        CGAL::pca_estimate_normals<CGAL::Parallel_if_available_tag>(point_set, 12);
        CGAL::mst_orient_normals(point_set, 12);
    }
    
    // Print parameters
    std::cout << "\n--- Parameters ---" << std::endl;
    std::cout << "  probability: " << probability << std::endl;
    std::cout << "  min_points: " << min_points << std::endl;
    std::cout << "  epsilon: " << epsilon << std::endl;
    std::cout << "  cluster_epsilon: " << cluster_epsilon << std::endl;
    std::cout << "  normal_threshold: " << normal_threshold << " degrees" << std::endl;
    
    // Setup RANSAC
    Efficient_ransac ransac;
    ransac.set_input(point_set, point_set.point_map(), point_set.normal_map());
    
    ransac.add_shape_factory<Plane>();
    if (detect_all) {
        ransac.add_shape_factory<Cylinder>();
        ransac.add_shape_factory<Cone>();
        ransac.add_shape_factory<Sphere>();
        ransac.add_shape_factory<Torus>();
    }
    
    Efficient_ransac::Parameters parameters;
    parameters.probability = probability;
    parameters.min_points = min_points;
    parameters.epsilon = epsilon;
    parameters.cluster_epsilon = cluster_epsilon;
    parameters.normal_threshold = std::cos(normal_threshold * M_PI / 180.0);
    
    // Run detection
    std::cout << "\n--- Running RANSAC ---" << std::endl;
    ransac.detect(parameters);
    
    // Results
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Shapes detected: " << ransac.shapes().size() << std::endl;
    
    // Add colors
    auto red = point_set.add_property_map<unsigned char>("red", 128).first;
    auto green = point_set.add_property_map<unsigned char>("green", 128).first;
    auto blue = point_set.add_property_map<unsigned char>("blue", 128).first;
    
    srand(42);
    std::size_t total_assigned = 0;
    
    for (auto it = ransac.shapes().begin(); it != ransac.shapes().end(); ++it) {
        unsigned char r = rand() % 156 + 100;
        unsigned char g = rand() % 156 + 100;
        unsigned char b = rand() % 156 + 100;
        
        for (std::size_t idx : (*it)->indices_of_assigned_points()) {
            auto pt_it = point_set.begin();
            std::advance(pt_it, idx);
            red[*pt_it] = r;
            green[*pt_it] = g;
            blue[*pt_it] = b;
        }
        total_assigned += (*it)->indices_of_assigned_points().size();
    }
    
    std::cout << "Points assigned: " << total_assigned << " / " << point_set.size()
              << " (" << (100.0 * total_assigned / point_set.size()) << "%)" << std::endl;
    
    // Save
    std::ofstream out(output_file);
    CGAL::IO::set_ascii_mode(out);
    out << point_set;
    out.close();
    
    std::cout << "\nOutput saved: " << output_file << std::endl;
    
    return EXIT_SUCCESS;
}
