#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Point_set.h>

#include <boost/iterator/function_output_iterator.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using FT = typename Kernel::FT;
using Point_3 = typename Kernel::Point_3;
using Vector_3 = typename Kernel::Vector_3;

using Point_set = CGAL::Point_set_3<Point_3>;

using Neighbor_query = CGAL::Shape_detection::Point_set::K_neighbor_query_for_point_set<Point_set>;
using Region_type = CGAL::Shape_detection::Point_set::Least_squares_plane_fit_region_for_point_set<Point_set>;
using Sorting = CGAL::Shape_detection::Point_set::Least_squares_plane_fit_sorting_for_point_set<Point_set, Neighbor_query>;
using Region_growing = CGAL::Shape_detection::Region_growing<Neighbor_query, Region_type>;

void print_usage(const char* program_name) {
    std::cout << "\n=== CGAL Region Growing Plane Detection ===" << std::endl;
    std::cout << "\nUsage: " << program_name << " <input.ply> [options]\n" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -k <value>          K nearest neighbors (default: 12)" << std::endl;
    std::cout << "  -dist <value>       Max distance to plane (default: auto)" << std::endl;
    std::cout << "  -angle <value>      Max normal angle in degrees (default: 25)" << std::endl;
    std::cout << "  -minpts <value>     Min points per region (default: 50)" << std::endl;
    std::cout << "  -output <filename>  Output filename (default: rg_detected.ply)" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " building.ply -k 20 -minpts 30" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv) {
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
    
    std::string input_file = argv[1];
    std::string output_file = "rg_detected.ply";
    std::size_t k_neighbors = 12;
    FT max_distance = -1;
    FT max_angle = 25;
    std::size_t min_region_size = 50;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-k" && i + 1 < argc) {
            k_neighbors = std::stoi(argv[++i]);
        } else if (arg == "-dist" && i + 1 < argc) {
            max_distance = std::stod(argv[++i]);
        } else if (arg == "-angle" && i + 1 < argc) {
            max_angle = std::stod(argv[++i]);
        } else if (arg == "-minpts" && i + 1 < argc) {
            min_region_size = std::stoi(argv[++i]);
        } else if (arg == "-output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    Point_set point_set;
    if (!CGAL::IO::read_point_set(input_file, point_set)) {
        std::cerr << "Error: Cannot read " << input_file << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "\n=== CGAL Region Growing Plane Detection ===" << std::endl;
    std::cout << "\nInput: " << input_file << std::endl;
    std::cout << "Points loaded: " << point_set.size() << std::endl;
    
    CGAL::Bbox_3 bbox = CGAL::bbox_3(
        CGAL::make_transform_iterator_from_property_map(point_set.begin(), point_set.point_map()),
        CGAL::make_transform_iterator_from_property_map(point_set.end(), point_set.point_map()));
    
    FT diagonal = std::sqrt(
        (bbox.xmax() - bbox.xmin()) * (bbox.xmax() - bbox.xmin()) +
        (bbox.ymax() - bbox.ymin()) * (bbox.ymax() - bbox.ymin()) +
        (bbox.zmax() - bbox.zmin()) * (bbox.zmax() - bbox.zmin()));
    
    std::cout << "Bounding box diagonal: " << diagonal << std::endl;
    
    if (max_distance < 0) max_distance = diagonal * 0.01;
    
    if (!point_set.has_normal_map()) {
        std::cout << "\nEstimating normals..." << std::endl;
        point_set.add_normal_map();
        CGAL::pca_estimate_normals<CGAL::Parallel_if_available_tag>(point_set, k_neighbors);
        CGAL::mst_orient_normals(point_set, k_neighbors);
    }
    
    std::cout << "\n--- Parameters ---" << std::endl;
    std::cout << "  k_neighbors: " << k_neighbors << std::endl;
    std::cout << "  max_distance: " << max_distance << std::endl;
    std::cout << "  max_angle: " << max_angle << " degrees" << std::endl;
    std::cout << "  min_region_size: " << min_region_size << std::endl;
    
    Neighbor_query neighbor_query = CGAL::Shape_detection::Point_set::make_k_neighbor_query(
        point_set, CGAL::parameters::k_neighbors(k_neighbors));
    
    Sorting sorting = CGAL::Shape_detection::Point_set::make_least_squares_plane_fit_sorting(
        point_set, neighbor_query);
    sorting.sort();
    
    Region_type region_type = CGAL::Shape_detection::Point_set::make_least_squares_plane_fit_region(
        point_set,
        CGAL::parameters::
            maximum_distance(max_distance).
            maximum_angle(max_angle).
            minimum_region_size(min_region_size));
    
    Region_growing region_growing(point_set, sorting.ordered(), neighbor_query, region_type);
    
    auto red = point_set.add_property_map<unsigned char>("red", 128).first;
    auto green = point_set.add_property_map<unsigned char>("green", 128).first;
    auto blue = point_set.add_property_map<unsigned char>("blue", 128).first;
    
    std::cout << "\n--- Running Region Growing ---" << std::endl;
    
    std::size_t num_regions = 0;
    std::size_t total_assigned = 0;
    srand(42);
    
    region_growing.detect(
        boost::make_function_output_iterator(
            [&](const std::pair<Region_type::Primitive, std::vector<typename Point_set::Index>>& region) {
                unsigned char r = rand() % 156 + 100;
                unsigned char g = rand() % 156 + 100;
                unsigned char b = rand() % 156 + 100;
                
                for (auto idx : region.second) {
                    red[idx] = r;
                    green[idx] = g;
                    blue[idx] = b;
                }
                total_assigned += region.second.size();
                ++num_regions;
            }
        )
    );
    
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Regions detected: " << num_regions << std::endl;
    std::cout << "Points assigned: " << total_assigned << " / " << point_set.size()
              << " (" << (100.0 * total_assigned / point_set.size()) << "%)" << std::endl;
    
    std::ofstream out(output_file);
    CGAL::IO::set_ascii_mode(out);
    out << point_set;
    out.close();
    
    std::cout << "\nOutput saved: " << output_file << std::endl;
    
    return EXIT_SUCCESS;
}
