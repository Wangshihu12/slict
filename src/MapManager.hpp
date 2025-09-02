#include "utility.h"

/**
 * MapManager 类用于管理SLAM系统中的先验地图数据
 * 主要功能包括：加载先验地图、合并点云、创建surfel地图、发布可视化数据
 */
class MapManager
{

private:

    // ROS节点句柄，用于ROS通信
    ros::NodeHandlePtr nh_ptr;

    // 先验地图数据存储
    CloudXYZITPtr  pmSurfGlobal;    // 全局面特征点云
    CloudXYZITPtr  pmEdgeGlobal;    // 全局边缘特征点云
    CloudXYZITPtr  priorMap;        // 合并后的先验地图
    ros::Publisher priorMapPub;     // 先验地图发布器，用于可视化

    // UFO surfel地图类型定义
    using ufoSurfelMap = ufo::map::SurfelMap;
    ufoSurfelMap surfelMapSurf;     // 面特征的surfel地图
    ufoSurfelMap surfelMapEdge;     // 边缘特征的surfel地图

public:

    /**
     * 析构函数：清理资源
     */
   ~MapManager();
   
    /**
     * 构造函数：初始化MapManager并加载先验地图
     * @param nh_ptr_ ROS节点句柄指针，用于参数获取和话题发布
     */
    MapManager(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        // 步骤1: 初始化点云数据结构
        pmSurfGlobal = CloudXYZITPtr(new CloudXYZIT());  // 创建全局面特征点云容器
        pmEdgeGlobal = CloudXYZITPtr(new CloudXYZIT());  // 创建全局边缘特征点云容器
        priorMap     = CloudXYZITPtr(new CloudXYZIT());  // 创建合并后的先验地图容器

        // 步骤2: 初始化ROS发布器
        priorMapPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/priormap", 10);

        // 步骤3: 获取先验地图目录路径参数
        string prior_map_dir = "";
        nh_ptr->param("/prior_map_dir", prior_map_dir, string(""));

        // 步骤4: 读取先验地图的关键帧位姿日志
        string pmPose_ = prior_map_dir + "/kfpose6d.pcd";  // 构建位姿文件路径
        CloudPosePtr pmPose(new CloudPose());              // 创建位姿点云容器
        pcl::io::loadPCDFile<PointPose>(pmPose_, *pmPose); // 加载位姿数据

        // 获取关键帧数量并输出信息
        int PM_KF_COUNT = pmPose->size();
        printf("Prior map path %s. Num scans: %d\n", pmPose_.c_str(), pmPose->size());

        // 步骤5: 并行读取面特征和边缘特征点云数据
        deque<CloudXYZITPtr> pmSurf(PM_KF_COUNT);  // 面特征点云队列
        deque<CloudXYZITPtr> pmEdge(PM_KF_COUNT);  // 边缘特征点云队列
        #pragma omp parallel for num_threads(MAX_THREADS)  // 使用OpenMP并行加速
        for (int i = 0; i < PM_KF_COUNT; i++)
        {
            // 创建第i帧的面特征点云容器并加载数据
            pmSurf[i] = CloudXYZITPtr(new CloudXYZIT());
            string pmSurf_ = prior_map_dir + "/pointclouds/" + "KfSurfPcl_" + to_string(i) + ".pcd";
            pcl::io::loadPCDFile<PointXYZIT>(pmSurf_, *pmSurf[i]);

            // 创建第i帧的边缘特征点云容器并加载数据
            pmEdge[i] = CloudXYZITPtr(new CloudXYZIT());
            string pmEdge_ = prior_map_dir + "/pointclouds/" + "KfEdgePcl_" + to_string(i) + ".pcd";
            pcl::io::loadPCDFile<PointXYZIT>(pmEdge_, *pmEdge[i]);

            // printf("Reading scan %d.\n", i);  // 调试信息（已注释）
        }

        printf("Merging the scans.\n");

        // 步骤6: 合并所有关键帧的特征点云
        for (int i = 0; i < PM_KF_COUNT; i++)
        {
            *pmSurfGlobal += *pmSurf[i];  // 累加面特征点云到全局地图
            *pmEdgeGlobal += *pmEdge[i];  // 累加边缘特征点云到全局地图
        }

        // 步骤7: 为可视化准备和发布先验地图
        *priorMap = *pmSurfGlobal + *pmEdgeGlobal;  // 合并面特征和边缘特征
        {
            // 对合并后的点云进行均匀下采样以减少数据量
            pcl::UniformSampling<PointXYZIT> downsampler;
            downsampler.setRadiusSearch(0.1);  // 设置采样半径为0.1米
            downsampler.setInputCloud(priorMap);
            downsampler.filter(*priorMap);     // 执行下采样过滤
        }

        // 发布先验地图用于RViz可视化
        Util::publishCloud(priorMapPub, *priorMap, ros::Time::now(), "world");

        printf(KYEL "Surfelizing the scans.\n" RESET);

        // 步骤8: 创建UFO surfel地图结构
        double leaf_size = 0.1; // 网格大小，TODO: 改为参数配置
        
        // 为面特征点云创建surfel地图
        surfelMapSurf = ufoSurfelMap(leaf_size);
        insertCloudToSurfelMap(surfelMapSurf, *pmSurfGlobal);  // 将面特征点云插入surfel地图
        
        // 为边缘特征点云创建surfel地图
        surfelMapEdge = ufoSurfelMap(leaf_size);
        insertCloudToSurfelMap(surfelMapEdge, *pmEdgeGlobal);  // 将边缘特征点云插入surfel地图

        // 输出完成信息和统计数据
        printf(KGRN "Done. Surfmap: %d. Edgemap: %d\n" RESET, surfelMapSurf.size(), surfelMapEdge.size());
    }
};
