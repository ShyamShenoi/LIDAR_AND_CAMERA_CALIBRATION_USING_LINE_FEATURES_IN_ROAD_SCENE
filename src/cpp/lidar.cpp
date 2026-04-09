#include <iostream>
#include <string>
#include <vector>
#include <cmath> //
#include <limits>
#include <sstream>
#include <fstream> //

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

// Filters
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

// Segmentation
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// Features
#include <pcl/features/normal_3d.h>

// Visualization
#include <pcl/visualization/pcl_visualizer.h>

// Eigen
#include <Eigen/Dense>

#include <algorithm> // for std::max_element, std::min_element
#include <numeric>   // for std::accumulate

#include <yaml-cpp/yaml.h>

// ---------------------------------------
// Type definitions
// ---------------------------------------
typedef pcl::PointXYZI PointT;

// ---------------------------------------
// Declarations
// ---------------------------------------
struct Config
{
    bool do_downsampling;
    float voxel_leaf_size;

    float x_min, x_max, y_min, y_max, z_min, z_max;

    bool apply_z_rotation;
    float z_rotation_degs;

    bool use_angle_filter;
    float angle_min_degs, angle_max_degs;

    int stat_mean_k;
    double stat_std_mul;

    double lane_plane_distance_threshold;
    int lane_plane_max_iterations;

    float lane_intensity_threshold;

    bool use_lane_y_constraint;
    float lane_y_min, lane_y_max;

    bool use_lane_y_check;
    float lane_line_max_y_range;
    float lane_line_mean_y_min, lane_line_mean_y_max;
    float lane_max_vertical_variation;

    int lane_max_lines;
    float lane_line_distance_threshold;

    float lane_line_duplicate_angle_cos;
    float lane_line_duplicate_endpoint_thresh;

    double pole_plane_distance_threshold;
    int pole_plane_max_iterations;

    float cluster_tolerance;
    int cluster_min_size, cluster_max_size;

    float cylinder_distance_threshold;
    int cylinder_max_iterations;
    float cylinder_min_radius, cylinder_max_radius;
    float cylinder_expansion_factor;
    float cylinder_eps_angle;
    int cylinder_min_inliers;
    float pole_min_height, pole_min_vertical_ratio;

    float auto_percentile;

    static Config load(const std::string &path)
    {
        YAML::Node y = YAML::LoadFile(path);
        Config c;
        c.do_downsampling = y["do_downsampling"].as<bool>();
        c.voxel_leaf_size = y["voxel_leaf_size"].as<float>();

        c.x_min = y["x_min"].as<float>();
        c.x_max = y["x_max"].as<float>();
        c.y_min = y["y_min"].as<float>();
        c.y_max = y["y_max"].as<float>();
        c.z_min = y["z_min"].as<float>();
        c.z_max = y["z_max"].as<float>();

        c.apply_z_rotation = y["apply_z_rotation"].as<bool>();
        c.z_rotation_degs = y["z_rotation_degs"].as<float>();

        c.use_angle_filter = y["use_angle_filter"].as<bool>();
        c.angle_min_degs = y["angle_min_degs"].as<float>();
        c.angle_max_degs = y["angle_max_degs"].as<float>();

        c.stat_mean_k = y["stat_mean_k"].as<int>();
        c.stat_std_mul = y["stat_std_mul"].as<double>();

        c.lane_plane_distance_threshold = y["lane_plane_distance_threshold"].as<double>();
        c.lane_plane_max_iterations = y["lane_plane_max_iterations"].as<int>();

        c.lane_intensity_threshold = y["lane_intensity_threshold"].as<float>();

        c.use_lane_y_constraint = y["use_lane_y_constraint"].as<bool>();
        c.lane_y_min = y["lane_y_min"].as<float>();
        c.lane_y_max = y["lane_y_max"].as<float>();

        c.use_lane_y_check = y["use_lane_y_check"].as<bool>();
        c.lane_line_max_y_range = y["lane_line_max_y_range"].as<float>();
        c.lane_line_mean_y_min = y["lane_line_mean_y_min"].as<float>();
        c.lane_line_mean_y_max = y["lane_line_mean_y_max"].as<float>();
        c.lane_max_vertical_variation = y["lane_max_vertical_variation"].as<float>();

        c.lane_max_lines = y["lane_max_lines"].as<int>();
        c.lane_line_distance_threshold = y["lane_line_distance_threshold"].as<float>();

        c.lane_line_duplicate_angle_cos = y["lane_line_duplicate_angle_cos"].as<float>();
        c.lane_line_duplicate_endpoint_thresh = y["lane_line_duplicate_endpoint_thresh"].as<float>();

        c.pole_plane_distance_threshold = y["pole_plane_distance_threshold"].as<double>();
        c.pole_plane_max_iterations = y["pole_plane_max_iterations"].as<int>();

        c.cluster_tolerance = y["cluster_tolerance"].as<float>();
        c.cluster_min_size = y["cluster_min_size"].as<int>();
        c.cluster_max_size = y["cluster_max_size"].as<int>();

        c.cylinder_distance_threshold = y["cylinder_distance_threshold"].as<float>();
        c.cylinder_max_iterations = y["cylinder_max_iterations"].as<int>();
        c.cylinder_min_radius = y["cylinder_min_radius"].as<float>();
        c.cylinder_max_radius = y["cylinder_max_radius"].as<float>();
        c.cylinder_expansion_factor = y["cylinder_expansion_factor"].as<float>();
        c.cylinder_eps_angle = y["cylinder_eps_angle"].as<float>();
        c.cylinder_min_inliers = y["cylinder_min_inliers"].as<int>();
        c.pole_min_height = y["pole_min_height"].as<float>();
        c.pole_min_vertical_ratio = y["pole_min_vertical_ratio"].as<float>();

        c.auto_percentile = y["auto_percentile"].as<float>();

        return c;
    }
};

static Config Params;

struct LineEndpoints;
struct DetectedLine;

// 1) "LineEndpoints"
struct LineEndpoints
{
    Eigen::Vector3f pmin;
    Eigen::Vector3f pmax;
};

// 2) "DetectedLine"
struct DetectedLine
{
    pcl::ModelCoefficients coeff;
    LineEndpoints endpoints;
};

pcl::PointCloud<PointT>::Ptr loadCloud(const std::string &pcd_file);
void downsampleCloud(pcl::PointCloud<PointT>::Ptr &cloud_in, float leaf_size);
void applyPassThrough(pcl::PointCloud<PointT>::Ptr &cloud_in,
                      const std::string &field,
                      float min_val,
                      float max_val);
void removeOutliers(pcl::PointCloud<PointT>::Ptr &cloud_in, int mean_k, double std_mul);
void segmentPlane(const pcl::PointCloud<PointT>::Ptr &cloud_in,
                  double distance_threshold,
                  int max_iterations,
                  pcl::PointCloud<PointT>::Ptr &plane_inliers,
                  pcl::PointCloud<PointT>::Ptr &rest);
pcl::PointCloud<PointT>::Ptr detectLaneMarkings(const pcl::PointCloud<PointT>::Ptr &in_cloud,
                                                float intensity_threshold);
std::vector<pcl::PointCloud<PointT>::Ptr> extractClusters(
    const pcl::PointCloud<PointT>::Ptr &in_cloud,
    float cluster_tolerance,
    int min_size,
    int max_size,
    const std::string &save_folder = "");
bool isLikelyPoleCluster(const pcl::PointCloud<PointT>::Ptr &cluster,
                         float maxDim,
                         float minDim);
bool meetsMinimumHeight(const pcl::PointCloud<PointT>::Ptr &cloud_pole,
                        float minHeight);
pcl::PointCloud<PointT>::Ptr expandCylinderInliers(
    const pcl::PointCloud<PointT>::Ptr &in_cloud,
    const pcl::ModelCoefficients::Ptr &cyl_coeff,
    float expansion_factor);
LineEndpoints computeLineEndpointsFromInliers(const pcl::PointCloud<PointT>::Ptr &inliers,
                                              const pcl::ModelCoefficients::Ptr &line_coeffs);
bool isDuplicateLine(const DetectedLine &A, const DetectedLine &B);
std::vector<pcl::ModelCoefficients> detectLaneLines(
    pcl::PointCloud<PointT>::Ptr lane_markings,
    int max_lines,
    float distance_thresh);
pcl::PointCloud<PointT>::Ptr detectPolesIterative(
    const std::vector<pcl::PointCloud<PointT>::Ptr> &clusters);
void visualizeCombined(const pcl::PointCloud<PointT>::Ptr &base_cloud,
                       const pcl::PointCloud<PointT>::Ptr &special_cloud,
                       const std::string &title = "Lane+Pole Viewer");

// ----------------------------------------------------------------------
// We'll keep track of final lane/pole lines in global vectors
// so we can draw them and pick them with the mouse callback.
// ----------------------------------------------------------------------
static std::vector<LineEndpoints> g_finalLaneLines;
static std::vector<LineEndpoints> g_finalPoleLines;

struct ClickableLine
{
    LineEndpoints endpoints;
    std::string label; // "lane" or "pole"
};
static std::vector<ClickableLine> g_allClickableLines;
static std::string g_output_dir = ".";

// ----------------------------------------------------------------------
// Mouse callback
// ----------------------------------------------------------------------
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

static void pp_callback(const pcl::visualization::PointPickingEvent &event, void *cookie)
{
    int idx = event.getPointIndex();
    if (idx == -1)
    {
        std::cout << "[MouseClick] No valid point.\n";
        return;
    }

    float x, y, z;
    event.getPoint(x, y, z);
    Eigen::Vector3f picked(x, y, z);
    std::cout << "[MouseClick] Picked (" << x << ", " << y << ", " << z << ")\n";

    float minDist = std::numeric_limits<float>::max();
    int bestIdx = -1;

    for (int i = 0; i < (int)g_allClickableLines.size(); i++)
    {
        auto &L = g_allClickableLines[i];
        Eigen::Vector3f A = L.endpoints.pmin;
        Eigen::Vector3f B = L.endpoints.pmax;
        Eigen::Vector3f AB = B - A;
        Eigen::Vector3f AP = picked - A;
        float t = AB.dot(AP) / AB.dot(AB);
        t = std::max(0.f, std::min(1.f, t));
        Eigen::Vector3f closest = A + t * AB;
        float dist = (closest - picked).norm();
        if (dist < minDist)
        {
            minDist = dist;
            bestIdx = i;
        }
    }

    const float PICK_THRESH = 1.0f;
    if (bestIdx < 0 || minDist > PICK_THRESH)
    {
        std::cout << "  No lane/pole line near that point.\n";
        return;
    }

    auto &chosen = g_allClickableLines[bestIdx];
    std::cout << "  Found " << chosen.label << " line near that point.\n"
              << "    Pmin= " << chosen.endpoints.pmin.transpose() << "\n"
              << "    Pmax= " << chosen.endpoints.pmax.transpose() << "\n";

    std::string jsonPath = g_output_dir + "/3d_endpoints.json";

    // Read the existing JSON content
    std::ifstream inFile(jsonPath);
    std::string existingContent;
    if (inFile)
    {
        std::stringstream buffer;
        buffer << inFile.rdbuf();
        existingContent = buffer.str();
        inFile.close();
    }

    // Ensure the JSON is valid
    if (existingContent.empty() || existingContent == "[]")
    {
        existingContent = "["; // Start new JSON array
    }
    else
    {
        // Remove the last closing bracket "]"
        size_t lastBracket = existingContent.rfind(']');
        if (lastBracket != std::string::npos)
        {
            existingContent = existingContent.substr(0, lastBracket);
        }
    }

    // Open file for writing (overwrite mode)
    std::ofstream ofs(jsonPath, std::ios::trunc);
    if (!ofs)
    {
        std::cerr << "[Error] Cannot open 3d_endpoints.json for writing\n";
        return;
    }

    // Append new entry
    if (existingContent.length() > 1) // If there's already data, add a comma
    {
        ofs << existingContent << ",\n";
    }
    else
    {
        ofs << existingContent; // Keep existing [
    }

    // Add new JSON entry
    ofs << "  {\n"
        << "    \"type\": \"" << chosen.label << "\",\n"
        << "    \"pmin\": [" << chosen.endpoints.pmin.x() << ", "
        << chosen.endpoints.pmin.y() << ", "
        << chosen.endpoints.pmin.z() << "],\n"
        << "    \"pmax\": [" << chosen.endpoints.pmax.x() << ", "
        << chosen.endpoints.pmax.y() << ", "
        << chosen.endpoints.pmax.z() << "]\n"
        << "  }\n]"; // Close JSON array properly

    ofs.close();
    std::cout << "  Wrote endpoints to 3d_endpoints.json\n";
}

// ----------------------------------------------------------------------
// Normal Helper functions
// ----------------------------------------------------------------------
pcl::PointCloud<PointT>::Ptr loadCloud(const std::string &pcd_file)
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    if (pcl::io::loadPCDFile<PointT>(pcd_file, *cloud) == -1)
    {
        PCL_ERROR("Failed to read PCD file: %s\n", pcd_file.c_str());
        return nullptr;
    }
    std::cout << "Loaded cloud with " << cloud->size() << " points from " << pcd_file << "\n";
    return cloud;
}

void downsampleCloud(pcl::PointCloud<PointT>::Ptr &cloud_in, float leaf_size)
{
    if (!cloud_in || cloud_in->empty())
        return;
    if (!Params.do_downsampling)
    {
        std::cout << "[Downsampling] Skipped (flag disabled).\n";
        return;
    }
    pcl::VoxelGrid<PointT> voxel;
    voxel.setInputCloud(cloud_in);
    voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
    pcl::PointCloud<PointT>::Ptr cloud_downsampled(new pcl::PointCloud<PointT>);
    voxel.filter(*cloud_downsampled);
    std::cout << "[VoxelGrid] Original: " << cloud_in->size()
              << " -> Downsampled: " << cloud_downsampled->size() << "\n";
    cloud_in.swap(cloud_downsampled);
}

void applyPassThrough(pcl::PointCloud<PointT>::Ptr &cloud_in,
                      const std::string &field,
                      float min_val,
                      float max_val)
{
    if (!cloud_in || cloud_in->empty())
        return;
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(cloud_in);
    pass.setFilterFieldName(field);
    pass.setFilterLimits(min_val, max_val);
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    pass.filter(*filtered);
    cloud_in.swap(filtered);
    std::cout << "[PassThrough: " << field << "] after: " << cloud_in->size() << " points.\n";
}

void removeOutliers(pcl::PointCloud<PointT>::Ptr &cloud_in, int mean_k, double std_mul)
{
    if (!cloud_in || cloud_in->empty())
        return;
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud_in);
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh(std_mul);
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    sor.filter(*filtered);
    std::cout << "[StatOutRemoval] Before: " << cloud_in->size()
              << ", After: " << filtered->size() << "\n";
    cloud_in.swap(filtered);
}

void segmentPlane(const pcl::PointCloud<PointT>::Ptr &cloud_in,
                  double distance_threshold,
                  int max_iterations,
                  pcl::PointCloud<PointT>::Ptr &plane_inliers,
                  pcl::PointCloud<PointT>::Ptr &rest)
{
    if (!cloud_in || cloud_in->empty())
    {
        PCL_ERROR("Input cloud is empty. No plane segmentation done.\n");
        return;
    }
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_threshold);
    seg.setMaxIterations(max_iterations);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    seg.setInputCloud(cloud_in);
    seg.segment(*inliers, *coeffs);

    if (inliers->indices.empty())
    {
        std::cerr << "[segmentPlane] No plane found.\n";
        return;
    }

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud_in);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*plane_inliers);
    extract.setNegative(true);
    extract.filter(*rest);

    std::cout << "[PlaneSeg] distanceThresh=" << distance_threshold
              << ", plane_inliers=" << plane_inliers->size()
              << ", rest=" << rest->size() << "\n";
}

float computeOtsuThreshold(const pcl::PointCloud<PointT>::Ptr &cloud_ground)
{
    if (!cloud_ground || cloud_ground->empty())
        return Params.lane_intensity_threshold; // fallback

    // 1) Find min/max intensity
    auto minmax = std::minmax_element(
        cloud_ground->points.begin(), cloud_ground->points.end(),
        [](auto &a, auto &b)
        { return a.intensity < b.intensity; });
    float Imin = minmax.first->intensity;
    float Imax = minmax.second->intensity;

    const int bins = 256;
    std::vector<int> hist(bins, 0);
    float bin_width = (Imax - Imin) / bins;

    // 2) Build histogram
    for (auto &pt : cloud_ground->points)
    {
        int idx = std::min(bins - 1,
                           int((pt.intensity - Imin) / bin_width));
        hist[idx]++;
    }

    // 3) Total points
    int total = cloud_ground->points.size();

    // 4) Otsu’s method
    std::vector<float> prob(bins);
    for (int i = 0; i < bins; ++i)
        prob[i] = float(hist[i]) / total;

    float sumB = 0, wB = 0;
    float sum1 = 0;
    for (int i = 0; i < bins; ++i)
        sum1 += i * prob[i];

    float maxVar = 0;
    int thresholdBin = 0;

    for (int i = 0; i < bins; ++i)
    {
        wB += prob[i];
        if (wB == 0)
            continue;
        float wF = 1 - wB;
        if (wF == 0)
            break;

        sumB += i * prob[i];
        float mB = sumB / wB;
        float mF = (sum1 - sumB) / wF;

        // Between-class variance
        float varBetween = wB * wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar)
        {
            maxVar = varBetween;
            thresholdBin = i;
        }
    }

    float thresh = Imin + thresholdBin * bin_width;
    std::cout << "[AutoThresh] Otsu threshold = " << thresh << "\n";
    return thresh;
}

float computePercentileThreshold(
    const pcl::PointCloud<PointT>::Ptr &cloud,
    float percentile)
{
    if (!cloud || cloud->empty())
        return Params.lane_intensity_threshold;

    std::vector<float> I;
    I.reserve(cloud->size());
    for (auto &pt : cloud->points)
        I.push_back(pt.intensity);
    std::sort(I.begin(), I.end());
    int idx = std::min((int)I.size() - 1,
                       (int)(percentile / 100.0f * I.size()));
    return I[idx];
}

pcl::PointCloud<PointT>::Ptr detectLaneMarkings(const pcl::PointCloud<PointT>::Ptr &in_cloud,
                                                float intensity_threshold)
{
    pcl::PointCloud<PointT>::Ptr lane(new pcl::PointCloud<PointT>);
    if (!in_cloud || in_cloud->empty())
        return lane;

    for (const auto &pt : in_cloud->points)
    {
        if (pt.intensity >= intensity_threshold)
        {
            if (Params.use_lane_y_constraint)
            {
                if (pt.y >= Params.lane_y_min && pt.y <= Params.lane_y_max)
                    lane->push_back(pt);
            }
            else
            {
                lane->push_back(pt);
            }
        }
    }
    std::cout << "[LaneDetect] Found " << lane->size() << " points >= " << intensity_threshold
              << " intensity";
    if (Params.use_lane_y_constraint)
        std::cout << " and with y in [" << Params.lane_y_min << ", " << Params.lane_y_max << "]";
    std::cout << ".\n";
    return lane;
}

std::vector<pcl::PointCloud<PointT>::Ptr> extractClusters(
    const pcl::PointCloud<PointT>::Ptr &in_cloud,
    float cluster_tolerance,
    int min_size,
    int max_size,
    const std::string &save_folder)
{
    std::vector<pcl::PointCloud<PointT>::Ptr> clusters;
    if (!in_cloud || in_cloud->empty())
        return clusters;

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(in_cloud);

    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_size);
    ec.setMaxClusterSize(max_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(in_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    int c_id = 0;
    for (const auto &c_ind : cluster_indices)
    {
        pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>);
        cluster->reserve(c_ind.indices.size());
        for (int idx : c_ind.indices)
            cluster->push_back(in_cloud->points[idx]);

        if (!save_folder.empty())
        {
            std::stringstream ss;
            ss << save_folder << "/cluster_" << c_id++ << ".pcd";
            pcl::io::savePCDFileBinary(ss.str(), *cluster);
        }
        clusters.push_back(cluster);
    }
    std::cout << "[extractClusters] Found " << clusters.size() << " clusters.\n";
    return clusters;
}

bool isLikelyPoleCluster(const pcl::PointCloud<PointT>::Ptr &cluster,
                         float maxDim,
                         float minDim)
{
    if (!cluster || cluster->empty())
        return false;

    float min_x = std::numeric_limits<float>::max(), max_x = -std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max(), max_y = -std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max(), max_z = -std::numeric_limits<float>::max();

    for (const auto &pt : cluster->points)
    {
        if (pt.x < min_x)
            min_x = pt.x;
        if (pt.x > max_x)
            max_x = pt.x;
        if (pt.y < min_y)
            min_y = pt.y;
        if (pt.y > max_y)
            max_y = pt.y;
        if (pt.z < min_z)
            min_z = pt.z;
        if (pt.z > max_z)
            max_z = pt.z;
    }

    float dx = max_x - min_x;
    float dy = max_y - min_y;
    float dz = max_z - min_z;

    if (dx > maxDim || dy > maxDim)
        return false;

    if (dx < minDim && dy < minDim && dz < minDim)
        return false;

    return true;
}

bool meetsMinimumHeight(const pcl::PointCloud<PointT>::Ptr &cloud_pole,
                        float minHeight)
{
    if (!cloud_pole || cloud_pole->empty())
        return false;

    float min_z = std::numeric_limits<float>::max();
    float max_z = -std::numeric_limits<float>::max();

    for (const auto &pt : cloud_pole->points)
    {
        if (pt.z < min_z)
            min_z = pt.z;
        if (pt.z > max_z)
            max_z = pt.z;
    }
    return ((max_z - min_z) >= minHeight);
}

pcl::PointCloud<PointT>::Ptr expandCylinderInliers(
    const pcl::PointCloud<PointT>::Ptr &in_cloud,
    const pcl::ModelCoefficients::Ptr &cyl_coeff,
    float expansion_factor)
{
    if (!in_cloud || cyl_coeff->values.size() < 7)
        return pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());

    float x0 = cyl_coeff->values[0];
    float y0 = cyl_coeff->values[1];
    float z0 = cyl_coeff->values[2];
    float dx = cyl_coeff->values[3];
    float dy = cyl_coeff->values[4];
    float dz = cyl_coeff->values[5];
    float r = cyl_coeff->values[6];

    float axis_len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (axis_len < 1e-6)
        return pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());

    dx /= axis_len;
    dy /= axis_len;
    dz /= axis_len;

    float search_r = r * expansion_factor;
    pcl::PointCloud<PointT>::Ptr expanded(new pcl::PointCloud<PointT>);
    expanded->reserve(in_cloud->size());

    for (const auto &pt : in_cloud->points)
    {
        float vx = pt.x - x0;
        float vy = pt.y - y0;
        float vz = pt.z - z0;

        float proj = vx * dx + vy * dy + vz * dz;
        float px = vx - proj * dx;
        float py = vy - proj * dy;
        float pz = vz - proj * dz;
        float dist_sq = px * px + py * py + pz * pz;

        if (dist_sq <= search_r * search_r)
            expanded->push_back(pt);
    }
    return expanded;
}

LineEndpoints computeLineEndpointsFromInliers(const pcl::PointCloud<PointT>::Ptr &inliers,
                                              const pcl::ModelCoefficients::Ptr &line_coeffs)
{
    LineEndpoints endpoints;
    endpoints.pmin = Eigen::Vector3f::Zero();
    endpoints.pmax = Eigen::Vector3f::Zero();

    if (!inliers || inliers->empty())
        return endpoints;
    if (line_coeffs->values.size() < 6)
        return endpoints;

    Eigen::Vector3f p0(line_coeffs->values[0],
                       line_coeffs->values[1],
                       line_coeffs->values[2]);
    Eigen::Vector3f dir(line_coeffs->values[3],
                        line_coeffs->values[4],
                        line_coeffs->values[5]);
    dir.normalize();

    float tmin = std::numeric_limits<float>::max();
    float tmax = -std::numeric_limits<float>::max();

    for (const auto &pt : inliers->points)
    {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        float t = (p - p0).dot(dir);
        if (t < tmin)
            tmin = t;
        if (t > tmax)
            tmax = t;
    }
    endpoints.pmin = p0 + tmin * dir;
    endpoints.pmax = p0 + tmax * dir;
    return endpoints;
}

bool isDuplicateLine(const DetectedLine &A, const DetectedLine &B)
{
    Eigen::Vector3f dirA(A.coeff.values[3], A.coeff.values[4], A.coeff.values[5]);
    Eigen::Vector3f dirB(B.coeff.values[3], B.coeff.values[4], B.coeff.values[5]);
    dirA.normalize();
    dirB.normalize();
    float dot = dirA.dot(dirB);
    if (dot < Params.lane_line_duplicate_angle_cos)
        return false;

    float distPmin = (A.endpoints.pmin - B.endpoints.pmin).norm();
    float distPmax = (A.endpoints.pmax - B.endpoints.pmax).norm();

    if (distPmin < Params.lane_line_duplicate_endpoint_thresh ||
        distPmax < Params.lane_line_duplicate_endpoint_thresh)
    {
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------
// detectLaneLines
// ----------------------------------------------------------------------
std::vector<pcl::ModelCoefficients> detectLaneLines(
    pcl::PointCloud<PointT>::Ptr lane_markings,
    int max_lines,
    float distance_thresh)
{
    std::vector<DetectedLine> accepted_lines;
    if (!lane_markings || lane_markings->empty())
        return {};

    pcl::PointCloud<PointT>::Ptr cloud_work(new pcl::PointCloud<PointT>(*lane_markings));
    int lines_extracted = 0;
    int line_candidate_id = 0;

    while (lines_extracted < max_lines && cloud_work->size() > 20)
    {
        line_candidate_id++;
        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(distance_thresh);
        seg.setMaxIterations(1000);

        pcl::PointIndices::Ptr inliers_line(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coeff_line(new pcl::ModelCoefficients);
        seg.setInputCloud(cloud_work);
        seg.segment(*inliers_line, *coeff_line);

        if (inliers_line->indices.empty() || inliers_line->indices.size() < 10)
            break;

        pcl::PointCloud<PointT>::Ptr inlier_cloud(new pcl::PointCloud<PointT>);
        inlier_cloud->reserve(inliers_line->indices.size());
        float sum_y = 0.0f;
        float min_y = std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();

        for (int idx : inliers_line->indices)
        {
            const PointT &pt = cloud_work->points[idx];
            inlier_cloud->push_back(pt);
            sum_y += pt.y;
            if (pt.y < min_y)
                min_y = pt.y;
            if (pt.y > max_y)
                max_y = pt.y;
        }
        float mean_y = sum_y / inliers_line->indices.size();
        float y_range = max_y - min_y;

        bool discard_line = false;
        std::string discard_reason;

        if (Params.use_lane_y_check)
        {
            if (y_range > Params.lane_line_max_y_range)
            {
                discard_line = true;
                discard_reason = "y-range (" + std::to_string(y_range) + ") too large";
            }
            else if (mean_y < Params.lane_line_mean_y_min || mean_y > Params.lane_line_mean_y_max)
            {
                discard_line = true;
                discard_reason = "mean y out of range";
            }
        }

        LineEndpoints le = computeLineEndpointsFromInliers(inlier_cloud, coeff_line);
        float dz = std::fabs(le.pmax.z() - le.pmin.z());
        if (!discard_line && dz > Params.lane_max_vertical_variation)
        {
            discard_line = true;
            discard_reason = "vertical variation too large";
        }

        if (discard_line)
        {
            std::cout << "Discarding lane line candidate #" << line_candidate_id
                      << " due to " << discard_reason << "\n";

            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(cloud_work);
            extract.setIndices(inliers_line);
            extract.setNegative(true);
            pcl::PointCloud<PointT>::Ptr remainder(new pcl::PointCloud<PointT>);
            extract.filter(*remainder);
            cloud_work.swap(remainder);
            continue;
        }

        lines_extracted++;
        std::cout << "Lane Line " << lines_extracted << " endpoints:\n"
                  << "  Pmin= " << le.pmin.transpose()
                  << ", Pmax= " << le.pmax.transpose() << "\n";

        // remove inliers
        {
            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(cloud_work);
            extract.setIndices(inliers_line);
            extract.setNegative(true);
            pcl::PointCloud<PointT>::Ptr remainder(new pcl::PointCloud<PointT>);
            extract.filter(*remainder);
            cloud_work.swap(remainder);
        }

        DetectedLine dl;
        dl.coeff = *coeff_line;
        dl.endpoints = le;
        accepted_lines.push_back(dl);
    }

    // final merges
    std::vector<bool> used(accepted_lines.size(), false);
    for (size_t i = 0; i < accepted_lines.size(); i++)
    {
        if (used[i])
            continue;

        for (size_t j = i + 1; j < accepted_lines.size(); j++)
        {
            if (!used[j] && isDuplicateLine(accepted_lines[i], accepted_lines[j]))
            {
                std::cout << "Merging/skipping line " << (j + 1)
                          << " because it duplicates line " << (i + 1) << ".\n";
                used[j] = true;
            }
        }
    }

    std::vector<DetectedLine> finalDetected;
    finalDetected.reserve(accepted_lines.size());
    for (size_t i = 0; i < accepted_lines.size(); i++)
    {
        if (!used[i])
            finalDetected.push_back(accepted_lines[i]);
    }

    std::vector<pcl::ModelCoefficients> final_lines;
    final_lines.reserve(finalDetected.size());

    std::cout << "[detectLaneLines] Found " << finalDetected.size() << " lines.\n";
    std::cout << "  (Coefficients are [x0 y0 z0 dx dy dz], meaning:\n"
              << "   a point on line is (x0,y0,z0), direction is (dx,dy,dz) )\n";

    for (size_t i = 0; i < finalDetected.size(); i++)
    {
        final_lines.push_back(finalDetected[i].coeff);

        auto &c = finalDetected[i].coeff;
        auto &ep = finalDetected[i].endpoints;

        std::cout << "  Lane line " << (i + 1) << " Coeffs: ";
        for (float v : c.values)
            std::cout << v << " ";
        std::cout << "\n    Endpoints: Pmin= " << ep.pmin.transpose()
                  << ",  Pmax= " << ep.pmax.transpose() << "\n";

        // Also store for final draw & picking
        g_finalLaneLines.push_back(ep);

        ClickableLine cLine;
        cLine.endpoints = ep;
        cLine.label = "lane";
        g_allClickableLines.push_back(cLine);
    }
    return final_lines;
}

// ----------------------------------------------------------------------
// detectPolesIterative
// ----------------------------------------------------------------------
pcl::PointCloud<PointT>::Ptr detectPolesIterative(
    const std::vector<pcl::PointCloud<PointT>::Ptr> &clusters)
{
    pcl::PointCloud<PointT>::Ptr all_poles(new pcl::PointCloud<PointT>);
    int cluster_id = 0;

    for (auto &cluster : clusters)
    {
        cluster_id++;
        if (!isLikelyPoleCluster(cluster, 1.0f, 0.05f))
            continue;

        float min_x = std::numeric_limits<float>::max(), max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max(), max_y = -std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max(), max_z = -std::numeric_limits<float>::max();

        for (const auto &pt : cluster->points)
        {
            if (pt.x < min_x)
                min_x = pt.x;
            if (pt.x > max_x)
                max_x = pt.x;
            if (pt.y < min_y)
                min_y = pt.y;
            if (pt.y > max_y)
                max_y = pt.y;
            if (pt.z < min_z)
                min_z = pt.z;
            if (pt.z > max_z)
                max_z = pt.z;
        }

        float horiz = std::sqrt((max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y));
        float vert = max_z - min_z;
        if (horiz < 1e-3)
            horiz = 1e-3;
        float ratio = vert / horiz;
        if (ratio < Params.pole_min_vertical_ratio)
            continue;

        pcl::PointCloud<PointT>::Ptr working(new pcl::PointCloud<PointT>(*cluster));
        bool found_cylinder = false;
        int cylinder_count = 0;

        while (!found_cylinder && working->size() > 30)
        {
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
            pcl::NormalEstimation<PointT, pcl::Normal> ne;
            ne.setInputCloud(working);
            ne.setSearchMethod(tree);
            ne.setKSearch(35);

            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            ne.compute(*normals);

            pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_CYLINDER);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
            seg.setEpsAngle(Params.cylinder_eps_angle);
            seg.setNormalDistanceWeight(0.05);
            seg.setMaxIterations(Params.cylinder_max_iterations);
            seg.setDistanceThreshold(Params.cylinder_distance_threshold);
            seg.setRadiusLimits(Params.cylinder_min_radius, Params.cylinder_max_radius);
            seg.setInputCloud(working);
            seg.setInputNormals(normals);

            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
            seg.segment(*inliers, *coeffs);

            if (inliers->indices.empty() || (int)inliers->indices.size() < Params.cylinder_min_inliers)
            {
                break;
            }
            else
            {
                cylinder_count++;
                auto expanded_inliers = expandCylinderInliers(working, coeffs, Params.cylinder_expansion_factor);
                pcl::PointIndices::Ptr expanded_idx(new pcl::PointIndices);
                expanded_idx->indices.reserve(expanded_inliers->size());

                for (std::size_t i = 0; i < working->size(); ++i)
                {
                    const auto &ptA = working->points[i];
                    bool match = false;
                    for (const auto &ptB : expanded_inliers->points)
                    {
                        if ((ptA.x == ptB.x) && (ptA.y == ptB.y) && (ptA.z == ptB.z))
                        {
                            match = true;
                            break;
                        }
                    }
                    if (match)
                        expanded_idx->indices.push_back(static_cast<int>(i));
                }

                pcl::PointCloud<PointT>::Ptr cloud_pole(new pcl::PointCloud<PointT>);
                {
                    pcl::ExtractIndices<PointT> extract;
                    extract.setInputCloud(working);
                    extract.setIndices(expanded_idx);
                    extract.setNegative(false);
                    extract.filter(*cloud_pole);
                }

                if (!meetsMinimumHeight(cloud_pole, Params.pole_min_height))
                {
                    std::cout << "[Cluster " << cluster_id << "] Found cylinder #"
                              << cylinder_count << " but it is below min height.\n";

                    pcl::ExtractIndices<PointT> ex;
                    ex.setInputCloud(working);
                    ex.setIndices(expanded_idx);
                    ex.setNegative(true);
                    pcl::PointCloud<PointT>::Ptr remainder(new pcl::PointCloud<PointT>);
                    ex.filter(*remainder);
                    working.swap(remainder);
                    continue;
                }

                LineEndpoints ep = computeLineEndpointsFromInliers(expanded_inliers, coeffs);
                std::cout << "[Cluster " << cluster_id << "] Cylinder #"
                          << cylinder_count << " endpoints:\n"
                          << "  Pmin=" << ep.pmin.transpose()
                          << ", Pmax=" << ep.pmax.transpose() << "\n";

                *all_poles += *cloud_pole;
                std::cout << "[Cluster " << cluster_id << "] Found cylinder #"
                          << cylinder_count << " with " << cloud_pole->size()
                          << " expanded points.\n";
                found_cylinder = true;

                g_finalPoleLines.push_back(ep);
                ClickableLine cLine;
                cLine.endpoints = ep;
                cLine.label = "pole";
                g_allClickableLines.push_back(cLine);
            }
        }
    }
    return all_poles;
}

// ----------------------------------------------------------------------
// [ADDED] Function to rotate the entire cloud by 'angle_degs' about Z-axis
// ----------------------------------------------------------------------
void rotateCloudZAxis(pcl::PointCloud<PointT>::Ptr &cloud_in, float angle_degs)
{
    if (!cloud_in || cloud_in->empty())
        return;

    float angle_rad = angle_degs * M_PI / 180.0f;
    Eigen::AngleAxisf rot(angle_rad, Eigen::Vector3f::UnitZ());

    for (auto &pt : cloud_in->points)
    {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        p = rot * p;
        pt.x = p.x();
        pt.y = p.y();
        pt.z = p.z();
    }
    std::cout << "[rotateCloudZAxis] Rotated cloud by " << angle_degs << " deg about Z.\n";
}

// ----------------------------------------------------------------------
// [ADDED] Function to filter by angle in the XY-plane (degrees).
//         Keeps points whose angle is in [angle_min, angle_max].
// ----------------------------------------------------------------------
void applyAnglePassThrough(pcl::PointCloud<PointT>::Ptr &cloud_in,
                           float angle_min_degs,
                           float angle_max_degs)
{
    if (!cloud_in || cloud_in->empty())
        return;

    // Convert degrees to radians
    float amin = angle_min_degs * M_PI / 180.0f;
    float amax = angle_max_degs * M_PI / 180.0f;

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    filtered->reserve(cloud_in->size());

    for (const auto &pt : cloud_in->points)
    {
        float angle = std::atan2(pt.y, pt.x);
        // If you want to handle wrap-around (e.g. from -179 to +179),
        // you can handle it carefully. For now, we assume straightforward in-range check.
        if (angle >= amin && angle <= amax)
        {
            filtered->push_back(pt);
        }
    }
    std::cout << "[applyAnglePassThrough] Kept " << filtered->size()
              << " of " << cloud_in->size() << " points, angle range ["
              << angle_min_degs << ", " << angle_max_degs << "] deg.\n";

    cloud_in.swap(filtered);
}

// ----------------------------------------------------------------------
// visualizeCombined
// ----------------------------------------------------------------------
void visualizeCombined(const pcl::PointCloud<PointT>::Ptr &base_cloud,
                       const pcl::PointCloud<PointT>::Ptr &special_cloud,
                       const std::string &title)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(title));
    viewer->setBackgroundColor(0.1, 0.1, 0.1);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> gray(base_cloud, 128, 128, 128);
    viewer->addPointCloud<PointT>(base_cloud, gray, "base_cloud");

    pcl::visualization::PointCloudColorHandlerCustom<PointT> red(special_cloud, 255, 0, 0);
    viewer->addPointCloud<PointT>(special_cloud, red, "special_cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "special_cloud");

    // Final lane lines in red, final pole lines in blue
    for (size_t i = 0; i < g_finalLaneLines.size(); i++)
    {
        auto &ep = g_finalLaneLines[i];
        std::stringstream ss;
        ss << "lane_line_" << i;
        viewer->addLine(
            pcl::PointXYZ(ep.pmin.x(), ep.pmin.y(), ep.pmin.z()),
            pcl::PointXYZ(ep.pmax.x(), ep.pmax.y(), ep.pmax.z()),
            1.0, 0.0, 0.0, // red
            ss.str());
    }
    for (size_t i = 0; i < g_finalPoleLines.size(); i++)
    {
        auto &ep = g_finalPoleLines[i];
        std::stringstream ss;
        ss << "pole_line_" << i;
        viewer->addLine(
            pcl::PointXYZ(ep.pmin.x(), ep.pmin.y(), ep.pmin.z()),
            pcl::PointXYZ(ep.pmax.x(), ep.pmax.y(), ep.pmax.z()),
            0.0, 0.0, 1.0, // blue
            ss.str());
    }

    // register picking callback
    viewer->registerPointPickingCallback(pp_callback);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(10);
    }
}

// ---------------------------------------
// Main
// ---------------------------------------
int main(int argc, char **argv)
{

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input_pcd> <config_yaml> [output_dir]\n";
        return -1;
    }

    std::string pcd_file = argv[1];
    std::string config_file = argv[2];
    
    if (argc >= 4) {
        g_output_dir = argv[3];
    }

    // Load all parameters from YAML:
    Params = Config::load(config_file);

    pcl::PointCloud<PointT>::Ptr cloud_in = loadCloud(pcd_file);
    if (!cloud_in || cloud_in->empty())
        return -1;

    // [ADDED] Optionally rotate the entire cloud to "align" the car for certain cameras.
    if (Params.apply_z_rotation)
    {
        rotateCloudZAxis(cloud_in, Params.z_rotation_degs);
    }

    downsampleCloud(cloud_in, Params.voxel_leaf_size);

    applyPassThrough(cloud_in, "x", Params.x_min, Params.x_max);
    applyPassThrough(cloud_in, "y", Params.y_min, Params.y_max);
    applyPassThrough(cloud_in, "z", Params.z_min, Params.z_max);

    // [ADDED] Optionally filter the cloud by angle in the XY-plane
    if (Params.use_angle_filter)
    {
        applyAnglePassThrough(cloud_in, Params.angle_min_degs, Params.angle_max_degs);
    }

    removeOutliers(cloud_in, Params.stat_mean_k, Params.stat_std_mul);

    // For lane pipeline
    pcl::PointCloud<PointT>::Ptr cloud_lane(new pcl::PointCloud<PointT>(*cloud_in));
    // For pole pipeline
    pcl::PointCloud<PointT>::Ptr cloud_pole(new pcl::PointCloud<PointT>(*cloud_in));

    // 1) Lane plane segmentation
    pcl::PointCloud<PointT>::Ptr lane_plane(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr lane_others(new pcl::PointCloud<PointT>);
    segmentPlane(cloud_lane, Params.lane_plane_distance_threshold,
                 Params.lane_plane_max_iterations, lane_plane, lane_others);
    if (!lane_plane->empty())
        pcl::io::savePCDFileBinary(
            g_output_dir + "/lane_plane_inliers.pcd",
            *lane_plane);

    // 2) Lane markings
    // pcl::PointCloud<PointT>::Ptr lane_markings =
    //    detectLaneMarkings(lane_plane, Params.lane_intensity_threshold);

    // float auto_thresh = computeOtsuThreshold(lane_plane);
    //  Compute both Otsu’s threshold and the 90th‐percentile threshold
    float T_otsu = computeOtsuThreshold(lane_plane);
    float T_perc = computePercentileThreshold(lane_plane, Params.auto_percentile);
    float auto_thresh = std::max(T_otsu, T_perc);
    std::cout << "[AutoThresh] Using max(Otsu=" << T_otsu
              << ", Perc" << Params.auto_percentile
              << "%=" << T_perc << ") = " << auto_thresh << "\n";

    pcl::PointCloud<PointT>::Ptr lane_markings = detectLaneMarkings(lane_plane, auto_thresh);

    if (!lane_markings->empty())
        pcl::io::savePCDFileBinary(
            g_output_dir + "/lane_markings.pcd",
            *lane_markings);

    // Clear JSON file at the start
    std::ofstream clearFile(g_output_dir + "/3d_endpoints.json", std::ios::trunc);
    clearFile << "[]"; // Start JSON array
    clearFile.close();

    // 3) Lane line detection
    auto lane_lines = detectLaneLines(lane_markings,
                                      Params.lane_max_lines,
                                      Params.lane_line_distance_threshold);

    // 4) Pole plane
    pcl::PointCloud<PointT>::Ptr pole_ground(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr pole_above(new pcl::PointCloud<PointT>);
    segmentPlane(cloud_pole, Params.pole_plane_distance_threshold,
                 Params.pole_plane_max_iterations, pole_ground, pole_above);
    if (!pole_ground->empty())
        pcl::io::savePCDFileBinary(
            g_output_dir + "/pole_ground_inliers.pcd",
            *pole_ground);
    if (!pole_above->empty())
        pcl::io::savePCDFileBinary(
            g_output_dir + "/pole_above_plane.pcd",
            *pole_above);

    // 5) Clusters & Poles
    auto clusters = extractClusters(
        pole_above,
        Params.cluster_tolerance,
        Params.cluster_min_size,
        Params.cluster_max_size,
        g_output_dir + "/clusters");

    pcl::PointCloud<PointT>::Ptr all_poles = detectPolesIterative(clusters);
    if (!all_poles->empty())
        pcl::io::savePCDFileBinary(
            g_output_dir + "/poles_detected.pcd",
            *all_poles);

    // Combine lane & poles
    pcl::PointCloud<PointT>::Ptr combined_red(new pcl::PointCloud<PointT>);
    *combined_red = *all_poles + *lane_markings;
    pcl::io::savePCDFileBinary(
        g_output_dir + "/poles_and_lane_markings.pcd",
        *combined_red);

    // Visualizeualize
    visualizeCombined(cloud_in, combined_red, "Lane + Pole Viewer");
    std::cout << "DONE\n";
    return 0;
}
