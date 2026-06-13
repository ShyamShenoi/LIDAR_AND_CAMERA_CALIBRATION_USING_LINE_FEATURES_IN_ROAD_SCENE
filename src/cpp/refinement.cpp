/******************************************************************************
 * Usage:
 *   ./refine <mask_path> <pcd_path> <intrinsic_json> <extrinsic_json>
 ******************************************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <random>
#include <ctime>
#include <cmath>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <jsoncpp/json/json.h>
#include <memory>

// --- PCL ---
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>

// --- OpenCV ---
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// --- Eigen ---
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <Eigen/Dense>

// ============================================================================

namespace conversions
{
    inline double angle_in_rads(double degrees)
    {
        return degrees * M_PI / 180.0;
    }
    static inline Eigen::Matrix4d GetDeltaT(const float var[6])
    {
        auto deltaR = Eigen::Matrix3d(
            Eigen::AngleAxisd(angle_in_rads(var[2]), Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(angle_in_rads(var[1]), Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(angle_in_rads(var[0]), Eigen::Vector3d::UnitX()));
        Eigen::Matrix4d deltaT = Eigen::Matrix4d::Identity();
        deltaT.block<3, 3>(0, 0) = deltaR;
        deltaT(0, 3) = var[3];
        deltaT(1, 3) = var[4];
        deltaT(2, 3) = var[5];
        return deltaT;
    }

    // Convert rotation matrix to Euler angles
    static inline Eigen::Vector3d convert_to_euler_angles(Eigen::Matrix3d rot)
    {
        float sy = sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
        bool singular = sy < 1e-6;
        float x, y, z;
        if (!singular)
        {
            x = atan2(rot(2, 1), rot(2, 2));
            y = atan2(-rot(2, 0), sy);
            z = atan2(rot(1, 0), rot(0, 0));
        }
        else
        {
            x = atan2(-rot(1, 2), rot(1, 1));
            y = atan2(-rot(2, 0), sy);
            z = 0;
        }
        Eigen::Vector3d eul;
        eul(0) = z;
        eul(1) = y;
        eul(2) = x;
        return eul;
    }
}

static void LoadIntrinsic(const std::string &filename, Eigen::Matrix3d &intrinsic,
                          cv::Mat &distortion)
{
    Json::Reader reader;
    Json::Value root;

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "Error Opening " << filename << std::endl;
        exit(1);
    }
    if (reader.parse(in, root, false))
    {
        auto name = root.getMemberNames();
        std::string id = *(name.begin());
        std::cout << id << std::endl;
        Json::Value data = root[id]["param"]["cam_K"]["data"];
        intrinsic << data[0][0].asDouble(), data[0][1].asDouble(),
            data[0][2].asDouble(), data[1][0].asDouble(), data[1][1].asDouble(),
            data[1][2].asDouble(), data[2][0].asDouble(), data[2][1].asDouble(),
            data[2][2].asDouble();

        Json::Value data_distort = root[id]["param"]["cam_dist"]["data"];
        int size = root[id]["param"]["cam_dist"]["cols"].asInt();
        std::vector<float> distortions;
        for (int i = 0; i < size; i++)
        {
            distortions.push_back(data_distort[0][i].asFloat());
        }
        distortion = cv::Mat(distortions, true);
    }
    in.close();
}

static void LoadExtrinsic(const std::string &extrinsic_json_path,
                          Eigen::Matrix4d &extrinsic)
{
    Json::Reader reader;
    Json::Value root;

    std::ifstream in(extrinsic_json_path, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "Error Opening " << extrinsic_json_path << std::endl;
        exit(1);
    }
    if (reader.parse(in, root, false))
    {
        auto name = root.getMemberNames();
        std::string id = *(name.begin());
        std::cout << id << std::endl;
        Json::Value data = root[id]["param"]["sensor_calib"]["data"];
        extrinsic << data[0][0].asDouble(), data[0][1].asDouble(),
            data[0][2].asDouble(), data[0][3].asDouble(), data[1][0].asDouble(),
            data[1][1].asDouble(), data[1][2].asDouble(), data[1][3].asDouble(),
            data[2][0].asDouble(), data[2][1].asDouble(), data[2][2].asDouble(),
            data[2][3].asDouble(), data[3][0].asDouble(), data[3][1].asDouble(),
            data[3][2].asDouble(), data[3][3].asDouble();
    }
    in.close();
}


Eigen::Matrix3d g_camera_intrinsic;
Eigen::Matrix4d g_extrinsic_current;
cv::Mat g_mask_image;
pcl::PointCloud<pcl::PointXYZI> g_lane_pole_cloud;
float g_current_cost_fun = 0.0f;
float g_current_fc_score = 0.0f;


float calculateCost(const float var[6], const cv::Mat *mask_image,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr lane_pole_cloud,
                    const Eigen::Matrix4d &extrinsic_current,
                    const Eigen::Matrix3d &camera_intrinsic)
{
    size_t pointCount = 0;
    Eigen::Matrix4d deltaT = conversions::GetDeltaT(var);
    Eigen::Matrix4d T = extrinsic_current * deltaT;

    for (const auto &pt : lane_pole_cloud->points)
    {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;
        Eigen::Vector4d v(pt.x, pt.y, pt.z, 1);
        Eigen::Vector4d cam_pt = T * v;
        if (cam_pt(2) <= 0)
            continue;
        Eigen::Vector3d proj = camera_intrinsic * cam_pt.head<3>();
        double inv_z = 1.0 / proj(2);
        int x = (int)std::round(proj(0) * inv_z);
        int y = (int)std::round(proj(1) * inv_z);
        if (x >= 0 && x < mask_image->cols && y >= 0 && y < mask_image->rows)
        {
            // For a binary mask, simply check if the pixel is white.
            cv::Vec3b color = mask_image->at<cv::Vec3b>(y, x);
            if (color[0] == 255 && color[1] == 255 && color[2] == 255)
            {
                pointCount++;
            }
        }
    }
    return (float)pointCount / lane_pole_cloud->points.size();
}

void random_search(int search_count, float delta_x, float delta_y, float delta_z,
                   float delta_roll, float delta_pitch, float delta_yaw)
{
    float var[6] = {0};
    float bestVal[6] = {0};
    float points_on_mask_counter = calculateCost(var, &g_mask_image,
                                                 g_lane_pole_cloud.makeShared(),
                                                 g_extrinsic_current, g_camera_intrinsic);
    float update_counter = 0.0f;

    std::default_random_engine generator(std::random_device{}());
    std::uniform_real_distribution<double> x_range(-delta_x, delta_x);
    std::uniform_real_distribution<double> y_range(-delta_y, delta_y);
    std::uniform_real_distribution<double> z_range(-delta_z, delta_z);
    std::uniform_real_distribution<double> roll_range(-delta_roll, delta_roll);
    std::uniform_real_distribution<double> pitch_range(-delta_pitch, delta_pitch);
    std::uniform_real_distribution<double> yaw_range(-delta_yaw, delta_yaw);

    for (int i = 0; i < search_count; i++)
    {
        var[0] = roll_range(generator);
        var[1] = pitch_range(generator);
        var[2] = yaw_range(generator);
        var[3] = x_range(generator);
        var[4] = y_range(generator);
        var[5] = z_range(generator);
        float cnt = calculateCost(var, &g_mask_image,
                                  g_lane_pole_cloud.makeShared(),
                                  g_extrinsic_current, g_camera_intrinsic);
        if (cnt > points_on_mask_counter)
        {
            update_counter++;
            points_on_mask_counter = cnt;
            for (size_t k = 0; k < 6; k++)
            {
                bestVal[k] = var[k];
            }
        }
    }
    g_extrinsic_current = g_extrinsic_current * conversions::GetDeltaT(bestVal);
    g_current_cost_fun = points_on_mask_counter;
    g_current_fc_score = 1.0f - (update_counter / (float)search_count);
}

void Save_Projection(const std::string img_name, const cv::Mat *mask_image,
                     const pcl::PointCloud<pcl::PointXYZI>::Ptr lane_pole_cloud)
{
    cv::Mat draw_img = mask_image->clone();
    Eigen::Matrix3d R = g_extrinsic_current.block<3, 3>(0, 0);
    Eigen::Vector3d t(g_extrinsic_current(0, 3),
                      g_extrinsic_current(1, 3),
                      g_extrinsic_current(2, 3));
    Eigen::Vector3d eul = conversions::convert_to_euler_angles(R);
    std::cout << "[Save_Projection] refined extrinsic:\n";
    //<< " R (roll/pitch/yaw) = " << eul(0) << " " << eul(1) << " " << eul(2)
    //<< "\n t = " << t(0) << " " << t(1) << " " << t(2) << "\n";

    for (const auto &pt : lane_pole_cloud->points)
    {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;
        Eigen::Vector4d v(pt.x, pt.y, pt.z, 1);
        Eigen::Vector4d cam_pt = g_extrinsic_current * v;
        if (cam_pt(2) <= 0)
            continue;
        Eigen::Vector3d proj = g_camera_intrinsic * cam_pt.head<3>();
        double inv_z = 1.0 / proj(2);
        int x = (int)std::round(proj(0) * inv_z);
        int y = (int)std::round(proj(1) * inv_z);
        if (x >= 0 && x < draw_img.cols && y >= 0 && y < draw_img.rows)
        {
            cv::circle(draw_img, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
        }
    }
    cv::imwrite(img_name, draw_img);
    std::cout << "Save_Projection wrote " << img_name << "\n";
}

void saveExtrinsicMatrix(const Eigen::Matrix4d &T, const std::string &filename)
{
    std::string save_path = filename + "_extrinsic.txt";
    std::ofstream file(save_path, std::ios::trunc);
    if (file.is_open())
    {
        file << std::fixed << std::setprecision(6) << T << std::endl;
        file.close();
        std::cout << "[saveExtrinsicMatrix saved to " << save_path << std::endl;
    }
    else
    {
        std::cerr << "[saveExtrinsicMatrix] unable to open " << save_path << std::endl;
    }
}

void calibrate(const cv::Mat *mask_image,
               const pcl::PointCloud<pcl::PointXYZI>::Ptr lane_pole_cloud,
               const std::string &image_filename)
{
    // Copy input data into global variables.
    g_mask_image = mask_image->clone();
    g_lane_pole_cloud = *lane_pole_cloud;

    float var[6] = {0}; // initial deltas
    float points_on_mask_counter = calculateCost(var, mask_image,
                                                 lane_pole_cloud,
                                                 g_extrinsic_current,
                                                 g_camera_intrinsic);

    std::cout << "Before refine: " << points_on_mask_counter << "\n";

    // start from
    /*
    for (size_t i = 0; i < 100; i++)
    {
        if (i % 2 == 0)
        {
        random_search(10000, 0.05, 0.05, 0.05, 1, 1, 1);
        flag1 = (g_current_fc_score == 1);
        }
        else
        {
        random_search(10000, 0.005, 0.005, 0.005, 0.5, 0.5, 0.5);
       flag2= (g_current_fc_score == 1);
        }
        if (flag1 && f2)
            break;
    }
    for (size_t i = 0; i < 100; i++)
    {
     random_search(1000, 0.005, 0.005, 0.005, 0.5, 0.5, 0.5);
    if (g_current_fc_score == 1)
          break;
    }
    for (size_t i = 0; i < 100; i++)
    {
    random_search(1000, 0.001, 0.001, 0.001, 0.1, 0.1, 0.1);
    if (g_current_fc_score == 1)
            break;
    }
    for (size_t i = 0; i < 100; i++)
    {
    random_search(1000, 0.001, 0.001, 0.001, 0.05, 0.05, 0.05);
    if (g_current_fc_score == 1)
            break;
    */
    // Random search loops with decreasing step sizes.
    bool flag1 = false, flag2 = false;
    for (size_t i = 0; i < 100; i++)
    {
        if (i % 2 == 0)
        {
            random_search(10000, 0.05, 0.05, 0.05, 1, 1, 1);
            flag1 = (g_current_fc_score == 1);
        }
        else
        {
            random_search(10000, 0.005, 0.005, 0.005, 0.5, 0.5, 0.5);
            flag2 = (g_current_fc_score == 1);
        }
        if (flag1 && flag2)
            break;
    }
    for (size_t i = 0; i < 100; i++)
    {
        random_search(1000, 0.005, 0.005, 0.005, 0.5, 0.5, 0.5);
        if (g_current_fc_score == 1)
            break;
    }
    for (size_t i = 0; i < 100; i++)
    {
        random_search(1000, 0.001, 0.001, 0.001, 0.1, 0.1, 0.1);
        if (g_current_fc_score == 1)
            break;
    }
    for (size_t i = 0; i < 100; i++)
    {
        random_search(1000, 0.001, 0.001, 0.001, 0.05, 0.05, 0.05);
        if (g_current_fc_score == 1)
            break;
    }

    std::cout << "After refine: " << g_current_cost_fun << "\n";

    
    std::string projection_filename = image_filename + "_feature_projection.png";
    Save_Projection(projection_filename, mask_image, lane_pole_cloud);
    saveExtrinsicMatrix(g_extrinsic_current, image_filename);
}


static std::string getFilenameWithoutExtension(const std::string &path)
{
    size_t last_slash = path.find_last_of("/\\");
    size_t last_dot = path.find_last_of(".");
    if (last_dot == std::string::npos || last_dot < last_slash)
        return path.substr(last_slash + 1);
    return path.substr(last_slash + 1, last_dot - (last_slash + 1));
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <mask_path> <pcd_path> <intrinsic_json> <extrinsic_json>\n"
                  << "Example:\n  " << argv[0]
                  << " data/mask.jpg data/calib.pcd data/intrinsic.json data/extrinsic.json\n";
        return 1;
    }
    std::string mask_path = argv[1];
    std::string pcd_path = argv[2];
    std::string intrinsic_json = argv[3];
    std::string extrinsic_json = argv[4];

   
    cv::Mat mask_img = cv::imread(mask_path, cv::IMREAD_COLOR);
    if (mask_img.empty())
    {
        std::cerr << "Cannot load mask " << mask_path << std::endl;
        return 1;
    }
    std::cout << "[Main] Loaded mask: " << mask_path
              << " size=" << mask_img.cols << "x" << mask_img.rows << "\n";

   
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *cloud) == -1)
    {
        std::cerr << "Error: could not load PCD from " << pcd_path << std::endl;
        return 1;
    }
    std::cout << "[Main] Loaded PCD: " << pcd_path
              << ", points=" << cloud->size() << "\n";

    
    Eigen::Matrix3d intrinsic;
    cv::Mat distortion;
    LoadIntrinsic(intrinsic_json, intrinsic, distortion);
    g_camera_intrinsic = intrinsic;

    
    Eigen::Matrix4d extrinsic;
    LoadExtrinsic(extrinsic_json, extrinsic);
    g_extrinsic_current = extrinsic;

    std::cout << "[Main] Refinement started...\n";

    
    std::string image_filename = getFilenameWithoutExtension(mask_path);
    calibrate(&mask_img, cloud, image_filename);

    std::cout << "its Done..\n";
    return 0;
}
