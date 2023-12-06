#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <map_generator/dynamic_obs.h> // 自定义消息类型
#include <geometry_msgs/Vector3.h>
#include <visualization_msgs/Marker.h>
#include <pcl_ros/point_cloud.h>
//pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>

double r_thr_ ;
double h_thr_;  
double v_thr_ ;
double hmin_; 
double hmax_; 
int minclustering_points_, maxclustering_points_;
int frame_; 
ros::Subscriber cloud_sub;
ros::Publisher  dynobs_pub;
ros::Publisher  marker_pub;

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {  

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (!cloud || cloud->points.empty()) {
        return;
    }
    //pass through filter,过滤地面和天花板
    for (const auto& point : cloud->points) {
        if (point.z >= hmin_ && point.z <= hmax_) {
            filtered_cloud->points.push_back(point);
        }
    }
    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = false;

    // 构建搜索树
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(filtered_cloud);

    // DBSCAN聚类
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.5);  // 邻近点的距离阈值
    ec.setMinClusterSize(minclustering_points_);        // 聚类的最小点数
    ec.setMaxClusterSize(maxclustering_points_);     // 聚类的最大点数
    ec.setSearchMethod(tree);
    ec.setInputCloud(filtered_cloud);
    ec.extract(cluster_indices);

    int id = 0;
    // 遍历聚类
    for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);

        for (const auto& idx : it->indices) {
            cloud_cluster->push_back((*filtered_cloud)[idx]);  // 添加点到聚类
        }

        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // 检查过滤后的聚类大小
        if (!cloud_cluster->empty()) {
            // 手动计算聚类的中心点
            Eigen::Vector4f centroid(0, 0, 0, 0);
            for (const auto& pt : cloud_cluster->points) {
                centroid[0] += pt.x;
                centroid[1] += pt.y;
                centroid[2] += pt.z;
            }
            centroid /= cloud_cluster->points.size();

            // 计算聚类的最小和最大值
            Eigen::Vector4f min_pt(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX), max_pt(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
            for (const auto& pt : cloud_cluster->points) {
                min_pt[0] = std::min(min_pt[0], pt.x);
                min_pt[1] = std::min(min_pt[1], pt.y);
                min_pt[2] = std::min(min_pt[2], pt.z);

                max_pt[0] = std::max(max_pt[0], pt.x);
                max_pt[1] = std::max(max_pt[1], pt.y);
                max_pt[2] = std::max(max_pt[2], pt.z);
            }

            // 创建一个Marker表示圆柱
            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";  // 或您点云数据的坐标系
            marker.header.stamp = ros::Time::now();
            marker.ns = "clusters";
            marker.id = id++;
            marker.type = visualization_msgs::Marker::CYLINDER;
            marker.action = visualization_msgs::Marker::ADD;

            // 设置圆柱的位置
            marker.pose.position.x = centroid[0];
            marker.pose.position.y = centroid[1];
            marker.pose.position.z = centroid[2];

            // 设置圆柱的尺寸
            float height = max_pt[2] - min_pt[2];
            float diameter = sqrt(pow(max_pt[0] - min_pt[0], 2) + pow(max_pt[1] - min_pt[1], 2));
            marker.scale.x = diameter;  // 圆柱直径
            marker.scale.y = diameter;
            marker.scale.z = height;    // 圆柱高度

            // 设置颜色
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
            marker.color.a = 0.5;

        if (height <= h_thr_ && diameter <= r_thr_) {
                marker_pub.publish(marker);
            }

        }
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "dbscan");
    ros::NodeHandle nh;
    nh.param("r_thr", r_thr_, 1.5);
    nh.param("h_thr", h_thr_, 5.0);
    nh.param("v_thr", v_thr_, 0.1);
    nh.param("minclustering_points", minclustering_points_, 100);
    nh.param("maxclustering_points", maxclustering_points_, 5000);
    nh.param("frame", frame_, 3);
    nh.param("hmin", hmin_, 0.2);
    nh.param("hmax", hmax_, 2.0);

    cloud_sub = nh.subscribe("/my_map/cloud_ob", 10, cloudCallback);
    dynobs_pub = nh.advertise<map_generator::dynamic_obs>("/dynamic_obs_local", 10);
    marker_pub = nh.advertise<visualization_msgs::Marker>("dynobs_vis", 10);

    ROS_INFO("DBSCAN START.");
    ros::spin();
    return 0;
}
