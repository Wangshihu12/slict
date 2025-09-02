/**
 * This file is part of slict.
 *
 * Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot
 * sg>, School of EEE Nanyang Technological Univertsity, Singapore
 *
 * For more information please see <https://britsknguyen.github.io>.
 * or <https://github.com/brytsknguyen/slict>.
 * If you use this code, please cite the respective publications as
 * listed on the above websites.
 *
 * slict is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * slict is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with slict.  If not, see <http://www.gnu.org/licenses/>.
 */

//
// Created by Thien-Minh Nguyen on 01/08/22.
//

#include <filesystem>
// #include <boost/format.hpp>
#include <condition_variable>
#include <deque>
#include <thread>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ceres/ceres.h>
#include <cv_bridge/cv_bridge.h>

/* All needed for kdtree of custom point type----------*/
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
/* All needed for kdtree of custom point type----------*/

/* All needed for filter of custom point type----------*/
#include <pcl/pcl_base.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/filters/filter.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/impl/crop_box.hpp>
/* All needed for filter of custom point type----------*/

// Basalt
#include "basalt/spline/se3_spline.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"
#include "basalt/spline/posesplinex.h"

// ROS
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/Imu.h"
#include <tf2_ros/static_transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include "tf/transform_broadcaster.h"
#include "slict/globalMapsPublish.h"

// UFO
#include <ufo/map/code/code_unordered_map.h>
#include <ufo/map/point_cloud.h>
#include <ufo/map/surfel_map.h>
#include <ufomap_msgs/UFOMapStamped.h>
#include <ufomap_msgs/conversions.h>
#include <ufomap_ros/conversions.h>

// Factor
#include "PoseLocalParameterization.h"
// #include "PreintBase.h"
// #include "factor/PreintFactor.h"
// #include "factor/PointToPlaneDisFactorCT.h"
#include "factor/RelOdomFactor.h"
// #include "factor/PoseFactorCT.h"
#include "factor/PoseAnalyticFactor.h"
#include "factor/PointToPlaneAnalyticFactor.hpp"
#include "factor/GyroAcceBiasAnalyticFactor.h"
#include "factor/VelocityAnalyticFactor.h"

// Custom for package
#include "utility.h"
#include "slict/FeatureCloud.h"
#include "slict/OptStat.h"
#include "slict/TimeLog.h"
#include "CloudMatcher.hpp"

// Add ikdtree
#include <ikdTree/ikd_Tree.h>

// myGN solver
#include "tmnSolver.h"
#include "factor/GyroAcceBiasFactorTMN.hpp"
#include "factor/Point2PlaneFactorTMN.hpp"

#include "PointToMapAssoc.h"

// #include "MapManager.hpp"

// #define SPLINE_N 4

using namespace std;
using namespace Eigen;
using namespace pcl;
using namespace basalt;

// Shorthands for ufomap
namespace ufopred     = ufo::map::predicate;
using ufoSurfelMap    = ufo::map::SurfelMap;
using ufoSurfelMapPtr = boost::shared_ptr<ufoSurfelMap>;
using ufoNode         = ufo::map::NodeBV;
using ufoSphere       = ufo::geometry::Sphere;
using ufoPoint3       = ufo::map::Point3;
//Create a prototype for predicates
auto PredProto = ufopred::HasSurfel()
              && ufopred::DepthMin(1)
              && ufopred::DepthMax(1)
              && ufopred::NumSurfelPointsMin(1)
              && ufopred::SurfelPlanarityMin(0.2);
// Declare the type of the predicate as a new type
typedef decltype(PredProto) PredType;

typedef Sophus::SE3d SE3d;

class Estimator
{

private:

    // Node handler
    ros::NodeHandlePtr nh_ptr;
    ros::Time program_start_time;
    
    // The coordinate frame at the initial position of the slam
    string slam_ref_frame = "world";
    // The coordinate frame that states on the sliding window is using
    string current_ref_frame = slam_ref_frame;

    bool autoexit = false;
    bool show_report = true;

    // Subscribers
    ros::Subscriber data_sub;

    // Service
    ros::ServiceServer global_maps_srv;       // For requesting the global map to be published

    // Synchronized data buffer
    mutex packet_buf_mtx;
    deque<slict::FeatureCloud::ConstPtr> packet_buf;

    bool ALL_INITED  = false;
    int  WINDOW_SIZE = 4;
    int  N_SUB_SEG   = 4;

    Vector3d GRAV = Vector3d(0, 0, 9.82);

    // Spline representing the trajectory
    // using PoseSpline = basalt::Se3Spline<SPLINE_N>;
    using PoseSplinePtr = std::shared_ptr<PoseSplineX>;
    PoseSplinePtr GlobalTraj = nullptr;
    
    int    SPLINE_N       = 4;
    double deltaT         = 0.05;
    double start_fix_span = 0.05;
    double final_fix_span = 0.05;

    int reassociate_steps = 0;
    int reassoc_rate = 3;
    vector<int> deskew_method = {0, 0};

    bool use_ceres = true;

    // Custom solver
    tmnSolver* mySolver = NULL;

    // Sliding window data (prefixed "Sw" )
    struct TimeSegment
    {
        TimeSegment(double start_time_, double final_time_)
            : start_time(start_time_), final_time(final_time_)
        {};

        double dt()
        {
            return (final_time - start_time);
        }

        double start_time;
        double final_time;
    };
    deque<deque<TimeSegment>> SwTimeStep;
    deque<CloudXYZITPtr>      SwCloud;
    deque<CloudXYZIPtr>       SwCloudDsk;
    deque<CloudXYZIPtr>       SwCloudDskDS;
    deque<vector<LidarCoef>>  SwLidarCoef;
    deque<map<int, int>>      SwDepVsAssoc;
    deque<deque<ImuSequence>> SwImuBundle;      // ImuSample defined in utility.h
    deque<deque<ImuProp>>     SwPropState;

    // Check list for adjusting the computation
    map<int, int> DVA;
    int total_lidar_coef;

    // States at the segments' start and final times; ss: segment's start time, sf: segment's final time
    deque<deque<Quaternd>> ssQua, sfQua;
    deque<deque<Vector3d>> ssPos, sfPos;
    deque<deque<Vector3d>> ssVel, sfVel;
    deque<deque<Vector3d>> ssBia, sfBia;
    deque<deque<Vector3d>> ssBig, sfBig;

    // IMU weight
    double GYR_N = 10;
    double GYR_W = 10;
    double ACC_N = 0.5;
    double ACC_W = 10;

    double ACC_SCALE = 1.0;

    // Velocity weight
    double POSE_N = 5;

    // Velocity weight
    double VEL_N = 10;

    // Lidar weight
    double lidar_weight = 10;

    Vector3d BG_BOUND = Vector3d(0.1, 0.1, 0.1);
    Vector3d BA_BOUND = Vector3d(0.1, 0.1, 0.2);

    int last_fixed_knot = 0;
    int first_fixed_knot = 0;

    double leaf_size = 0.1;
    double assoc_spacing = 0.2;
    int surfel_map_depth = 5;
    int surfel_min_point = 5;
    int surfel_min_depth = 0;
    int surfel_query_depth = 3;
    double surfel_intsect_rad = 0.5;
    double surfel_min_plnrty = 0.8;

    PredType *commonPred;

    // Size of k-nearest neighbourhood for the knn search
    double dis_to_surfel_max = 0.05;
    double score_min = 0.1;
    
    // Lidar downsample rate
    int lidar_ds_rate = 1;
    int sweep_len = 1;

    // Optimization parameters
    double lidar_loss_thres = 1.0;
    double imu_loss_thres = -1.0;

    // Keeping track of preparation before solving
    TicToc tt_preopt;
    TicToc tt_fitspline;
    double t_slv_budget;

    // Solver config
    ceres::LinearSolverType linSolver;
    ceres::TrustRegionStrategyType trustRegType;     // LEVENBERG_MARQUARDT, DOGLEG
    ceres::DenseLinearAlgebraLibraryType linAlgbLib; // EIGEN, LAPACK, CUDA
    double max_solve_time = 0.5;
    int max_iterations = 200;
    bool ensure_real_time = true;
    bool find_factor_cost = false;
    bool fit_spline = false;

    // Sensors used
    bool fuse_lidar      = true;
    bool fuse_imu        = true;
    bool fuse_poseprop   = true;
    bool fuse_velprop    = true;
    
    bool snap_to_0180    = false;
    bool regularize_imu  = true;
    bool lite_redeskew   = false;
    int  fix_mode        = 1;
    double imu_init_time = 0.1;
    int max_outer_iters  = 1;
    double dj_thres      = 0.1;
    int max_lidar_factor = 5000;

    // Map
    CloudPosePtr        KfCloudPose;
    deque<CloudXYZIPtr> KfCloudinB;
    deque<CloudXYZIPtr> KfCloudinW;

    bool refine_kf = false;

    int    ufomap_version = 0;
    mutex  global_map_mtx;
    CloudXYZIPtr globalMap;

    TicToc tt_margcloud;
    TicToc tt_ufoupdate;

    mutex map_mtx;
    ufoSurfelMapPtr activeSurfelMap;
    ikdtreePtr activeikdtMap;

    mutex mapqueue_mtx;
    deque<CloudXYZIPtr> mapqueue;
    thread thread_update_map;

    ufoSurfelMapPtr priorSurfelMapPtr;
    ufoSurfelMap priorSurfelMap;

    ikdtreePtr priorikdtMapPtr;
    // ikdtree priorikdtMap;

    enum RelocStat {NOT_RELOCALIZED, RELOCALIZING, RELOCALIZED};

    RelocStat reloc_stat = NOT_RELOCALIZED;
    mutex relocBufMtx;
    deque<myTf<double>> relocBuf;
    ros::Subscriber relocSub;
    mytf tf_Lprior_L0;
    // mytf tf_Lprior_L0_init;         // For debugging and development only
    bool refine_reloc_tf = false;
    int  ioa_max_iter = 20;
    bool marginalize_new_points = false;

    thread reloc_init;

    // Loop closure
    bool loop_en = true;
    int loop_kf_nbr = 5;            // Number of neighbours to check for loop closure
    int loop_time_mindiff = 10;     // Only check for loop when keyframes have this much difference
    struct LoopPrior
    {
        LoopPrior(int prevPoseId_, int currPoseId_, double JKavr_, double IcpFn_, mytf tf_Bp_Bc_)
            : prevPoseId(prevPoseId_), currPoseId(currPoseId_), JKavr(JKavr_), IcpFn(IcpFn_), tf_Bp_Bc(tf_Bp_Bc_) {};

        int prevPoseId = -1;
        int currPoseId = -1;
        double JKavr = -1;
        double IcpFn = -1;
        mytf tf_Bp_Bc;
    };
    deque<LoopPrior> loopPairs;     // Array to store loop priors

    int icpMaxIter = 20;            // Maximum iterations for ICP
    double icpFitnessThres = 0.3;   // Fitness threshold for ICP check
    double histDis = 15.0;          // Maximum correspondence distance for icp
    double lastICPFn = -1;

    int rib_edge = 5;
    double odom_q_noise = 0.1;
    double odom_p_noise = 0.1;
    double loop_weight  = 0.02;

    TicToc tt_loopBA;               // Timer to check the loop and BA time
    struct BAReport
    {
        int turn = -1;
        double pgopt_time = 0;
        int pgopt_iter = 0;
        int factor_relpose = 0;
        int factor_loop = 0;
        double J0 = 0;
        double JK = 0;
        double J0_relpose = 0;
        double JK_relpose = 0;
        double J0_loop = 0;
        double JK_loop = 0;
        double rebuildmap_time = 0;
    };
    BAReport baReport;

    struct KeyframeCand
    {
        KeyframeCand(double start_time_, double end_time_, CloudXYZIPtr kfCloud_)
            : start_time(start_time_), end_time(end_time_), kfCloud(kfCloud_) {};
        double start_time;
        double end_time;
        CloudXYZIPtr kfCloud;
    };
    
    // Publisher for global map.
    ros::Publisher global_map_pub;
    bool publish_map = false;

    // Keyframe params
    double kf_min_dis = 0.5;
    double kf_min_angle = 10;
    double margPerc = 0;
    
    // Publisher for latest keyframe
    ros::Publisher kfcloud_pub;
    ros::Publisher kfcloud_std_pub;
    ros::Publisher kfpose_pub;

    // Log
    string log_dir = "/home/tmn";
    string log_dir_kf;
    std::ofstream loop_log_file;

    // PriorMap
    bool use_prior_map = false;

    std::thread initPriorMapThread;

    // For visualization
    CloudXYZIPtr  pmSurfGlobal;
    CloudXYZIPtr  pmEdgeGlobal;

    CloudXYZIPtr  priorMap;
    
    deque<CloudXYZIPtr> pmFull;
    CloudPosePtr pmPose;

    bool pmLoaded = false;

    // KdTreeFLANN<PointXYZI>::Ptr kdTreePriorMap;

    ros::Publisher priorMapPub;
    ros::Timer     pmVizTimer;

    double priormap_viz_res = 0.2;
    
    // ufoSurfelMap surfelMapSurf;
    // ufoSurfelMap surfelMapEdge;

    // Use ufomap or ikdtree;
    bool use_ufm = false;
    
public:
    // Destructor
    ~Estimator() {}

    Estimator(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {   
        // Normal Processes
        Initialize();

        // Inialize Prior map
        InitializePriorMap();

    }

    void Initialize()
    {

        program_start_time = ros::Time::now();

        autoexit = GetBoolParam("/autoexit", false);

        // Disable report
        show_report = GetBoolParam("/show_report", true);

        // Get the coordinate frame of choice
        nh_ptr->getParam("/slam_ref_frame", slam_ref_frame);
        current_ref_frame = slam_ref_frame;

        // Maximum number of threads
        printf("Maximum number of threads: %d\n", MAX_THREADS);

        // Window size length
        if (nh_ptr->getParam("/WINDOW_SIZE", WINDOW_SIZE))
            printf("WINDOW_SIZE declared: %d\n", WINDOW_SIZE);
        else
        {
            printf("WINDOW_SIZE not found. Exiting\n");
            exit(-1);
        }

        if (nh_ptr->getParam("/N_SUB_SEG", N_SUB_SEG))
            printf("N_SUB_SEG declared: %d\n", N_SUB_SEG);
        else
        {
            printf("N_SUB_SEG not found. Exiting\n");
            exit(-1);
        }

        nh_ptr->param("/SPLINE_N", SPLINE_N, 4);
        nh_ptr->param("/deltaT", deltaT, 0.05);
        nh_ptr->param("/start_fix_span", start_fix_span, 0.05);
        nh_ptr->param("/final_fix_span", final_fix_span, 0.05);
        nh_ptr->param("/reassociate_steps", reassociate_steps, 0);
        nh_ptr->param("/deskew_method", deskew_method, deskew_method);
        nh_ptr->param("/reassoc_rate", reassoc_rate, 3);

        use_ceres = GetBoolParam("/use_ceres", false);

        // Initialize the states in the sliding window
        ssQua = sfQua = deque<deque<Quaternd>>(WINDOW_SIZE, deque<Quaternd>(N_SUB_SEG, Quaternd::Identity()));
        ssPos = sfPos = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, Vector3d(0, 0, 0)));
        ssVel = sfVel = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, Vector3d(0, 0, 0)));
        ssBia = sfBia = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, Vector3d(0, 0, 0)));
        ssBig = sfBig = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, Vector3d(0, 0, 0)));

        // Gravity constant
        double GRAV_ = 9.82;
        nh_ptr->param("/GRAV", GRAV_, 9.82);
        GRAV = Vector3d(0, 0, GRAV_);
        printf("GRAV constant: %f\n", GRAV_);

        nh_ptr->getParam("/GYR_N", GYR_N);
        nh_ptr->getParam("/GYR_W", GYR_W);
        nh_ptr->getParam("/ACC_N", ACC_N);
        nh_ptr->getParam("/ACC_W", ACC_W);

        printf("Gyro variance: %f\n", GYR_N);
        printf("Bgyr variance: %f\n", GYR_W);
        printf("Acce variance: %f\n", ACC_N);
        printf("Bacc variance: %f\n", ACC_W);

        nh_ptr->getParam("/POSE_N", POSE_N);
        printf("Position prior variance: %f\n", POSE_N);

        nh_ptr->getParam("/VEL_N", VEL_N);
        printf("Velocity variance: %f\n", VEL_N);

        // Sensor weightage
        nh_ptr->getParam("/lidar_weight", lidar_weight);

        // Bias bounds
        vector<double> BG_BOUND_ = {0.1, 0.1, 0.1};
        vector<double> BA_BOUND_ = {0.1, 0.1, 0.2};
        nh_ptr->getParam("/BG_BOUND", BG_BOUND_);
        nh_ptr->getParam("/BA_BOUND", BA_BOUND_);
        BG_BOUND = Vector3d(BG_BOUND_[0], BG_BOUND_[1], BG_BOUND_[2]);
        BA_BOUND = Vector3d(BA_BOUND_[0], BA_BOUND_[1], BA_BOUND_[2]);

        // If use ufm for incremental map
        use_ufm = GetBoolParam("/use_ufm", false);

        // Downsample size
        nh_ptr->getParam("/leaf_size",          leaf_size);
        nh_ptr->getParam("/assoc_spacing",      assoc_spacing);
        nh_ptr->getParam("/surfel_map_depth",   surfel_map_depth);
        nh_ptr->getParam("/surfel_min_point",   surfel_min_point);
        nh_ptr->getParam("/surfel_min_depth",   surfel_min_depth);
        nh_ptr->getParam("/surfel_query_depth", surfel_query_depth);
        nh_ptr->getParam("/surfel_intsect_rad", surfel_intsect_rad);
        nh_ptr->getParam("/surfel_min_plnrty",  surfel_min_plnrty);

        printf("leaf_size:          %f\n", leaf_size);
        printf("assoc_spacing:      %f\n", assoc_spacing);
        printf("surfel_map_depth:   %d\n", surfel_map_depth);
        printf("surfel_min_point:   %d\n", surfel_min_point);
        printf("surfel_min_depth:   %d\n", surfel_min_depth);
        printf("surfel_query_depth: %d\n", surfel_query_depth);
        printf("surfel_intsect_rad: %f\n", surfel_intsect_rad);
        printf("surfel_min_plnrty:  %f\n", surfel_min_plnrty);

        commonPred = new PredType(ufopred::HasSurfel()
                               && ufopred::DepthMin(surfel_min_depth)
                               && ufopred::DepthMax(surfel_query_depth - 1)
                               && ufopred::NumSurfelPointsMin(surfel_min_point)
                               && ufopred::SurfelPlanarityMin(surfel_min_plnrty));

        // Number of neigbours to check for in association
        nh_ptr->getParam("/dis_to_surfel_max", dis_to_surfel_max);
        nh_ptr->getParam("/score_min", score_min);
        // Lidar feature downsample rate
        // nh_ptr->getParam("/ds_rate", ds_rate);
        // Lidar sweep len by number of scans merged
        nh_ptr->getParam("/sweep_len", sweep_len);

        // Keyframe params
        nh_ptr->getParam("/kf_min_dis", kf_min_dis);
        nh_ptr->getParam("/kf_min_angle", kf_min_angle);
        // Refine keyframe with ICP
        refine_kf = GetBoolParam("/refine_kf", false);

        // Optimization parameters
        nh_ptr->getParam("/lidar_loss_thres", lidar_loss_thres);

        // Solver
        string linSolver_;
        nh_ptr->param("/linSolver", linSolver_, string("dqr"));
        if (linSolver_ == "dqr")
            linSolver = ceres::DENSE_QR;
        else if( linSolver_ == "dnc")
            linSolver = ceres::DENSE_NORMAL_CHOLESKY;
        else if( linSolver_ == "snc")
            linSolver = ceres::SPARSE_NORMAL_CHOLESKY;
        else if( linSolver_ == "cgnr")
            linSolver = ceres::CGNR;
        else if( linSolver_ == "dschur")
            linSolver = ceres::DENSE_SCHUR;
        else if( linSolver_ == "sschur")
            linSolver = ceres::SPARSE_SCHUR;
        else if( linSolver_ == "ischur")
            linSolver = ceres::ITERATIVE_SCHUR;
        else
            linSolver = ceres::SPARSE_NORMAL_CHOLESKY;
        printf(KYEL "/linSolver: %d. %s\n" RESET, linSolver, linSolver_.c_str());

        string trustRegType_;
        nh_ptr->param("/trustRegType", trustRegType_, string("lm"));
        if (trustRegType_ == "lm")
            trustRegType = ceres::LEVENBERG_MARQUARDT;
        else if( trustRegType_ == "dogleg")
            trustRegType = ceres::DOGLEG;
        else
            trustRegType = ceres::LEVENBERG_MARQUARDT;
        printf(KYEL "/trustRegType: %d. %s\n" RESET, trustRegType, trustRegType_.c_str());

        string linAlgbLib_;
        nh_ptr->param("/linAlgbLib", linAlgbLib_, string("cuda"));
        if (linAlgbLib_ == "eigen")
            linAlgbLib = ceres::DenseLinearAlgebraLibraryType::EIGEN;
        else if(linAlgbLib_ == "lapack")
            linAlgbLib = ceres::DenseLinearAlgebraLibraryType::LAPACK;
        else if(linAlgbLib_ == "cuda")
            linAlgbLib = ceres::DenseLinearAlgebraLibraryType::CUDA;
        else
            linAlgbLib = ceres::DenseLinearAlgebraLibraryType::EIGEN;
        printf(KYEL "/linAlgbLib: %d. %s\n" RESET, linAlgbLib, linAlgbLib_.c_str());

        nh_ptr->param("/max_solve_time", max_solve_time,  0.5);
        nh_ptr->param("/max_iterations", max_iterations,  200);
        
        ensure_real_time = GetBoolParam("/ensure_real_time", true);
        find_factor_cost = GetBoolParam("/find_factor_cost", true);
        fit_spline       = GetBoolParam("/fit_spline", true);

        // Fusion option
        fuse_lidar     = GetBoolParam("/fuse_lidar",     true);
        fuse_imu       = GetBoolParam("/fuse_imu",       true);
        fuse_poseprop  = GetBoolParam("/fuse_poseprop",  true);
        fuse_velprop   = GetBoolParam("/fuse_velprop",   true);

        snap_to_0180   = GetBoolParam("/snap_to_0180",   false);
        regularize_imu = GetBoolParam("/regularize_imu", true);
        lite_redeskew  = GetBoolParam("/lite_redeskew",  false);

        nh_ptr->param("/fix_mode",         fix_mode,         1);
        nh_ptr->param("/imu_init_time",    imu_init_time,    0.1);
        nh_ptr->param("/max_outer_iters",  max_outer_iters,  1);
        nh_ptr->param("/max_lidar_factor", max_lidar_factor, 4000);
        nh_ptr->param("/dj_thres",         dj_thres,         0.1);

        printf("max_outer_iters: %d.\n"
               "dj_thres:        %f.\n" 
               "fix_mode:        %d.\n"
               "max_iterations:  %d.\n"
               "imu_init_time:   %f\n",
                max_outer_iters, dj_thres, fix_mode, max_iterations, imu_init_time);
        
        // Loop parameters
        loop_en = GetBoolParam("/loop_en", true);
        nh_ptr->param("/loop_kf_nbr", loop_kf_nbr, 5);
        nh_ptr->param("/loop_time_mindiff", loop_time_mindiff, 10);

        nh_ptr->param("/icpMaxIter", icpMaxIter, 20);
        nh_ptr->param("/icpFitnessThres", icpFitnessThres, 0.3);
        nh_ptr->param("/histDis", histDis, 15.0);

        nh_ptr->param("/rib_edge", rib_edge, 5);
        nh_ptr->param("/odom_q_noise", odom_q_noise, 0.1);
        nh_ptr->param("/odom_p_noise", odom_p_noise, 0.1);
        nh_ptr->param("/loop_weight", loop_weight, 0.1);
        
        // Map inertialization
        KfCloudPose   = CloudPosePtr(new CloudPose());

        // Create a handle to the global map
        activeSurfelMap = ufoSurfelMapPtr(new ufoSurfelMap(leaf_size, surfel_map_depth));
        // Create an ikdtree
        activeikdtMap = ikdtreePtr(new ikdtree(0.5, 0.6, leaf_size));

        // For visualization
        globalMap = CloudXYZIPtr(new CloudXYZI());

        // Advertise the global map
        global_map_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/global_map", 10);

        publish_map = GetBoolParam("/publish_map", true);

        // Subscribe to the lidar-imu package
        data_sub = nh_ptr->subscribe("/sensors_sync", 100, &Estimator::DataHandler, this);

        // Advertise the outputs
        kfcloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/kfcloud", 10);
        kfpose_pub  = nh_ptr->advertise<sensor_msgs::PointCloud2>("/kfpose", 10);

        kfcloud_std_pub = nh_ptr->advertise<slict::FeatureCloud>("/kfcloud_std", 10);

        // Advertise the service
        global_maps_srv = nh_ptr->advertiseService("/global_maps_publish", &Estimator::PublishGlobalMaps, this);

        // Log file
        log_dir = nh_ptr->param("/log_dir", log_dir);
        log_dir_kf = log_dir + "/KFCloud/";
        std::filesystem::create_directories(log_dir);
        std::filesystem::create_directories(log_dir_kf);

        loop_log_file.open(log_dir + "/loop_log.csv");
        loop_log_file.precision(std::numeric_limits<double>::digits10 + 1);
        // loop_log_file.close();

        // Create a prior map manager
        // mapManager = new MapManager(nh_ptr);

        // Create a thread to update the map
        thread_update_map = thread(&Estimator::UpdateMap, this); ;
    }

    void InitializePriorMap()
    {
        TicToc tt_initprior;

        tf_Lprior_L0 = myTf(Quaternd(1, 0, 0, 0), Vector3d(0, 0, 0));

        use_prior_map = GetBoolParam("/use_prior_map", false);
        printf("use_prior_map:   %d\n", use_prior_map);

        if (!use_prior_map)
            return;

        // Check if initial pose is available
        if(ros::param::has("/tf_Lprior_L0_init"))
        {
            vector<double> tf_Lprior_L0_init_;
            nh_ptr->param("/tf_Lprior_L0_init", tf_Lprior_L0_init_, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            printf(KYEL "tf_Lprior_L0_init: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n" RESET,
                         tf_Lprior_L0_init_[0], tf_Lprior_L0_init_[1], tf_Lprior_L0_init_[2],
                         tf_Lprior_L0_init_[3], tf_Lprior_L0_init_[4], tf_Lprior_L0_init_[5]);
            
            myTf tf_Lprior_L0_init(Util::YPR2Quat(tf_Lprior_L0_init_[3], tf_Lprior_L0_init_[4], tf_Lprior_L0_init_[5]),
                                   Vector3d(tf_Lprior_L0_init_[0], tf_Lprior_L0_init_[1], tf_Lprior_L0_init_[2]));
            
            // lock_guard<mutex>lg(relocBufMtx);
            // relocBuf.push_back(tf_Lprior_L0_init);

            geometry_msgs::PoseStamped reloc_pose;
            reloc_pose.header.stamp = ros::Time::now();
            reloc_pose.header.frame_id = "priormap";
            reloc_pose.pose.position.x = tf_Lprior_L0_init.pos(0);
            reloc_pose.pose.position.y = tf_Lprior_L0_init.pos(1);
            reloc_pose.pose.position.z = tf_Lprior_L0_init.pos(2);
            reloc_pose.pose.orientation.x = tf_Lprior_L0_init.rot.x();
            reloc_pose.pose.orientation.y = tf_Lprior_L0_init.rot.y();
            reloc_pose.pose.orientation.z = tf_Lprior_L0_init.rot.z();
            reloc_pose.pose.orientation.w = tf_Lprior_L0_init.rot.w();

            reloc_init = std::thread(&Estimator::PublishManualReloc, this, reloc_pose);
        }

        // The name of the keyframe cloud
        string priormap_kfprefix = "cloud";
        nh_ptr->param("/priormap_kfprefix", priormap_kfprefix, string("cloud"));

        // Downsampling rate for visualizing the priormap
        nh_ptr->param("/priormap_viz_res", priormap_viz_res, 0.2);

        // Refine the relocalization transform
        refine_reloc_tf = GetBoolParam("/relocalization/refine_reloc_tf", false);
        marginalize_new_points = GetBoolParam("/relocalization/marginalize_new_points", false);

        // Get the maximum
        nh_ptr->param("/relocalization/ioa_max_iter", ioa_max_iter, 20);
        printf("ioa_max_iter: %d\n", ioa_max_iter);

        // Subscribe to the relocalization
        relocSub = nh_ptr->subscribe("/reloc_pose", 100, &Estimator::RelocCallback, this);

        // pmSurfGlobal = CloudXYZIPtr(new CloudXYZI());
        // pmEdgeGlobal = CloudXYZIPtr(new CloudXYZI());
        priorMap = CloudXYZIPtr(new CloudXYZI());

        // Initializing priormap
        priorMapPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/priormap", 10);

        string prior_map_dir = "";
        nh_ptr->param("/prior_map_dir", prior_map_dir, string(""));

        // Read the pose log of priormap
        string pmPose_ = prior_map_dir + "/kfpose6d.pcd";
        pmPose = CloudPosePtr(new CloudPose());
        pcl::io::loadPCDFile<PointPose>(pmPose_, *pmPose);

        int PM_KF_COUNT = pmPose->size();
        printf(KGRN "Prior map path %s. Num scans: %d. Begin loading ...\n" RESET, pmPose_.c_str(), pmPose->size());
        
        // pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
        if(use_ufm)
        {
            if (!std::filesystem::exists(prior_map_dir + "/ufo_surf_map.um"))
            {
                printf("Prebuilt UFO surf map not found, creating one.\n");

                // Reading the surf feature from log
                pmFull = deque<CloudXYZIPtr>(PM_KF_COUNT);
                #pragma omp parallel for num_threads(MAX_THREADS)
                for (int i = 0; i < PM_KF_COUNT; i++)
                {
                    pmFull[i] = CloudXYZIPtr(new CloudXYZI());
                    string pmFull_ = prior_map_dir + "/pointclouds/" + priormap_kfprefix + "_" + zeroPaddedString(i, PM_KF_COUNT) + ".pcd";
                    pcl::io::loadPCDFile<PointXYZI>(pmFull_, *pmFull[i]);

                    // Filter the pointclouds
                    CloudXYZIPtr temp(new CloudXYZI());
                    for(auto &point : pmFull[i]->points)
                    {
                        if (Util::PointIsValid(point) && Util::pointDistanceSq(point) != 0)
                            temp->push_back(point);
                        else
                            printf(KRED "Invalid points: %f, %f, %f\n" RESET, point.x, point.y, point.z);
                    }

                    printf("Reading scan %s. Valid: %7.3d / %7.3d (%4.0f%%).\n", 
                            pmFull_.c_str(), temp->size(), pmFull[i]->size(), temp->size()*100 / pmFull[i]->size());
                    pmFull[i] = temp;
                }

                pmLoaded = true;

                printf("\n");
                printf("Merging the scans:\n");

                Vector3d pmPointMin(FLT_MAX, FLT_MAX, FLT_MAX), pmPointMax(FLT_MIN, FLT_MIN, FLT_MIN);

                // Merge the scans
                for (int i = 0; i < PM_KF_COUNT; i++)
                {
                    *priorMap += *pmFull[i];

                    for(auto &point : pmFull[i]->points)
                    {
                        pmPointMin.x() = min(pmPointMin.x(), (double)point.x);
                        pmPointMin.y() = min(pmPointMin.y(), (double)point.y);
                        pmPointMin.z() = min(pmPointMin.z(), (double)point.z);

                        pmPointMax.x() = max(pmPointMax.x(), (double)point.x);
                        pmPointMax.y() = max(pmPointMax.y(), (double)point.y);
                        pmPointMax.z() = max(pmPointMax.z(), (double)point.z);                    
                    }

                    printf("Scan %s of %d points merged. Bounds: %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f\n",
                            zeroPaddedString(i, PM_KF_COUNT).c_str(), pmFull[i]->size(),
                            pmPointMin.x(), pmPointMin.y(), pmPointMin.z(),
                            pmPointMax.x(), pmPointMax.y(), pmPointMax.z());
                }

                printf("Surfelizing the prior map of %d points.\n", priorMap->size());

                priorSurfelMapPtr = ufoSurfelMapPtr(new ufoSurfelMap(leaf_size, surfel_map_depth));
                insertCloudToSurfelMap(*priorSurfelMapPtr, *priorMap);

                // Downsample the prior map for visualization in another thread
                auto pmVizFunctor = [this](const CloudXYZIPtr& priorMap_)->void
                {
                    // CloudXYZI priorMapDS;
                    pcl::UniformSampling<PointXYZI> downsampler;
                    downsampler.setRadiusSearch(priormap_viz_res);
                    downsampler.setInputCloud(priorMap);
                    downsampler.filter(*priorMap);

                    Util::publishCloud(priorMapPub, *priorMap, ros::Time::now(), "map");

                    if (!refine_reloc_tf)
                    {
                        for(auto &cloud : pmFull)
                            cloud->clear();
                    }

                    return;
                };

                initPriorMapThread = std::thread(pmVizFunctor, std::ref(priorMap));

                printf("Prior Map built %d. Save the prior map...\n", priorSurfelMapPtr->size());

                // Save the ufomap object
                priorSurfelMapPtr->write(prior_map_dir + "/ufo_surf_map.um");
            }
            else
            {
                printf("Prebuilt UFO surf map found, loading...\n");
                priorSurfelMapPtr = ufoSurfelMapPtr(new ufoSurfelMap(prior_map_dir + "/ufo_surf_map.um"));

                // Merge and downsample the prior map for visualization in another thread
                auto pmVizFunctor = [this, priormap_kfprefix](string prior_map_dir, int PM_KF_COUNT, CloudXYZIPtr& priorMap_)->void
                {
                    // Reading the surf feature from log
                    pmFull = deque<CloudXYZIPtr>(PM_KF_COUNT);
                    #pragma omp parallel for num_threads(MAX_THREADS)
                    for (int i = 0; i < PM_KF_COUNT; i++)
                    {
                        pmFull[i] = CloudXYZIPtr(new CloudXYZI());
                        string pmFull_ = prior_map_dir + "/pointclouds/" + priormap_kfprefix + "_" + zeroPaddedString(i, PM_KF_COUNT) + ".pcd";
                        pcl::io::loadPCDFile<PointXYZI>(pmFull_, *pmFull[i]);

                        printf("Reading scan %s.\n", pmFull_.c_str());
                    }

                    pmLoaded = true;

                    // Merge the scans
                    for (int i = 0; i < pmFull.size(); i++)
                    {
                        *priorMap += *pmFull[i];
                        // printf("Map size: %d\n", priorMap->size());
                    }

                    // CloudXYZI priorMapDS;
                    pcl::UniformSampling<PointXYZI> downsampler;
                    downsampler.setRadiusSearch(priormap_viz_res);
                    downsampler.setInputCloud(priorMap);
                    downsampler.filter(*priorMap);

                    Util::publishCloud(priorMapPub, *priorMap, ros::Time::now(), "map");

                    if (!refine_reloc_tf)
                    {
                        for(auto &cloud : pmFull)
                            cloud->clear();
                    }

                    return;
                };

                initPriorMapThread = std::thread(pmVizFunctor, prior_map_dir, PM_KF_COUNT, std::ref(priorMap));
            }
        }
        else
        {
            if (!std::filesystem::exists(prior_map_dir + "/priormap.pcd"))
            {
                printf("Prebuilt pcd not found, creating one.\n");

                // Reading the surf feature from log
                pmFull = deque<CloudXYZIPtr>(PM_KF_COUNT);
                #pragma omp parallel for num_threads(MAX_THREADS)
                for (int i = 0; i < PM_KF_COUNT; i++)
                {
                    pmFull[i] = CloudXYZIPtr(new CloudXYZI());
                    string pmFull_ = prior_map_dir + "/pointclouds/" + priormap_kfprefix + "_" + zeroPaddedString(i, PM_KF_COUNT) + ".pcd";
                    pcl::io::loadPCDFile<PointXYZI>(pmFull_, *pmFull[i]);

                    // Filter the pointclouds
                    CloudXYZIPtr temp(new CloudXYZI());
                    for(auto &point : pmFull[i]->points)
                    {
                        if (Util::PointIsValid(point) && Util::pointDistanceSq(point) != 0)
                            temp->push_back(point);
                        else
                            printf(KRED "Invalid points: %f, %f, %f\n" RESET, point.x, point.y, point.z);
                    }

                    printf("Reading scan %s. Valid: %7.3d / %7.3d (%4.0f%%).\n", 
                            pmFull_.c_str(), temp->size(), pmFull[i]->size(), temp->size()*100 / pmFull[i]->size());
                    pmFull[i] = temp;
                }

                printf("\n");
                printf("Merging the scans:\n");

                Vector3d pmPointMin(FLT_MAX, FLT_MAX, FLT_MAX), pmPointMax(FLT_MIN, FLT_MIN, FLT_MIN);

                // Merge the scans
                for (int i = 0; i < PM_KF_COUNT; i++)
                {
                    *priorMap += *pmFull[i];

                    for(auto &point : pmFull[i]->points)
                    {
                        pmPointMin.x() = min(pmPointMin.x(), (double)point.x);
                        pmPointMin.y() = min(pmPointMin.y(), (double)point.y);
                        pmPointMin.z() = min(pmPointMin.z(), (double)point.z);

                        pmPointMax.x() = max(pmPointMax.x(), (double)point.x);
                        pmPointMax.y() = max(pmPointMax.y(), (double)point.y);
                        pmPointMax.z() = max(pmPointMax.z(), (double)point.z);                    
                    }

                    printf("Scan %s of %d points merged. Bounds: %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f\n",
                            zeroPaddedString(i, PM_KF_COUNT).c_str(), pmFull[i]->size(),
                            pmPointMin.x(), pmPointMin.y(), pmPointMin.z(),
                            pmPointMax.x(), pmPointMax.y(), pmPointMax.z());
                }

                printf("Building ikdtree prior map of %d points.\n", priorMap->size());

                // CloudXYZI priorMapDS;
                pcl::UniformSampling<PointXYZI> downsampler;
                downsampler.setRadiusSearch(priormap_viz_res);
                downsampler.setInputCloud(priorMap);
                downsampler.filter(*priorMap);

                Util::publishCloud(priorMapPub, *priorMap, ros::Time::now(), "map");

                if (!refine_reloc_tf)
                {
                    for(auto &cloud : pmFull)
                        cloud->clear();
                }

                // Build the ikdtree
                priorikdtMapPtr = ikdtreePtr(new ikdtree(0.5, 0.6, leaf_size));
                priorikdtMapPtr->Build(priorMap->points);

                pmLoaded = true;

                printf("Prior Map built %d. Save the prior map...\n", priorikdtMapPtr->size());

                // Save the fullmap pcd
                pcl::io::savePCDFileBinary(prior_map_dir + "/priormap.pcd", *priorMap);
            }
            else
            {
                printf("Prebuilt pcd map found, loading...\n");

                // Load the map
                pcl::io::loadPCDFile<PointXYZI>(prior_map_dir + "/priormap.pcd", *priorMap);;

                // Build the ikdtree
                priorikdtMapPtr = ikdtreePtr(new ikdtree(0.5, 0.6, leaf_size));
                priorikdtMapPtr->Build(priorMap->points);
               
                // Merge and downsample the prior map for visualization in another thread
                auto pmVizFunctor = [this, priormap_kfprefix](string prior_map_dir, int PM_KF_COUNT, CloudXYZIPtr& priorMap_)->void
                {
                    // Reading the surf feature from log
                    pmFull = deque<CloudXYZIPtr>(PM_KF_COUNT);
                    #pragma omp parallel for num_threads(MAX_THREADS)
                    for (int i = 0; i < PM_KF_COUNT; i++)
                    {
                        pmFull[i] = CloudXYZIPtr(new CloudXYZI());
                        string pmFull_ = prior_map_dir + "/pointclouds/" + priormap_kfprefix + "_" + zeroPaddedString(i, PM_KF_COUNT) + ".pcd";
                        pcl::io::loadPCDFile<PointXYZI>(pmFull_, *pmFull[i]);

                        printf("Reading scan %s.\n", pmFull_.c_str());
                    }

                    pmLoaded = true;

                    // // Merge the scans
                    // for (int i = 0; i < pmFull.size(); i++)
                    // {
                    //     *priorMap += *pmFull[i];
                    //     // printf("Map size: %d\n", priorMap->size());
                    // }

                    // // CloudXYZI priorMapDS;
                    // pcl::UniformSampling<PointXYZI> downsampler;
                    // downsampler.setRadiusSearch(priormap_viz_res);
                    // downsampler.setInputCloud(priorMap);
                    // downsampler.filter(*priorMap);

                    Util::publishCloud(priorMapPub, *priorMap, ros::Time::now(), "map");

                    if (!refine_reloc_tf)
                    {
                        for(auto &cloud : pmFull)
                            cloud->clear();
                    }

                    return;
                };

                initPriorMapThread = std::thread(pmVizFunctor, prior_map_dir, PM_KF_COUNT, std::ref(priorMap));
            }
        }
        
        printf(KGRN "Done. Time: %f\n" RESET, tt_initprior.Toc());

        // pmVizTimer = nh_ptr->createTimer(ros::Duration(5.0), &Estimator::PublishPriorMap, this);
    }

    void PublishManualReloc(geometry_msgs::PoseStamped relocPose_)
    {
        ros::Publisher relocPub = nh_ptr->advertise<geometry_msgs::PoseStamped>("/reloc_pose", 100);
        geometry_msgs::PoseStamped relocPose = relocPose_;
        while(true)
        {
            if(reloc_stat != RELOCALIZED)
            {
                relocPub.publish(relocPose);
                printf("Manual reloc pose published.\n");
            }
            else
                break;

            this_thread::sleep_for(chrono::milliseconds(1000));
        }
    }

    void RelocCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        static bool one_shot = true;
        if (!one_shot)
            return;
        one_shot = false;

        if(reloc_stat == RELOCALIZED)
            return;

        myTf<double> tf_Lprior_L0(*msg);
        
        printf(KYEL "Received Reloc Pose: %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f\n" RESET,
                     tf_Lprior_L0.pos(0), tf_Lprior_L0.pos(1), tf_Lprior_L0.pos(2),
                     tf_Lprior_L0.yaw(), tf_Lprior_L0.pitch(), tf_Lprior_L0.roll());

        if(refine_reloc_tf)
        {
            reloc_stat = RELOCALIZING;

            while (!pmLoaded)
            {
                printf(KYEL "Waiting for prior map to be done\n" RESET);
                this_thread::sleep_for(chrono::milliseconds(100));
            }

            while(KfCloudPose->size() == 0)
            {
                printf(KYEL "Waiting for first keyframe to be made\n" RESET);
                this_thread::sleep_for(chrono::milliseconds(100));
            }

            // Search for closeby prior keyframes
            pcl::KdTreeFLANN<PointPose> kdTreePmPose;
            kdTreePmPose.setInputCloud(pmPose);

            int knn_nbrkf = min(5, (int)pmPose->size());
            vector<int> knn_idx(knn_nbrkf); vector<float> knn_sq_dis(knn_nbrkf);
            kdTreePmPose.nearestKSearch((tf_Lprior_L0*myTf(KfCloudPose->back())).Pose6D(), knn_nbrkf, knn_idx, knn_sq_dis);

            // Create a local map
            CloudXYZIPtr localMap(new CloudXYZI());
            for(int i = 0; i < knn_idx.size(); i++)
                *localMap += *pmFull[knn_idx[i]];

            // Create a cloud matcher
            CloudMatcher cm(0.1, 0.1);

            // Run ICP to find the relative pose
            // Matrix4f tfm_Lprior_L0;
            // double icpFitness = 0;
            // double icpTime = 0;
            // cm.CheckICP(localMap, KfCloudinW.back(), tf_Lprior_L0.cast<float>().tfMat(), tfm_Lprior_L0,
            //             10, 10, 1.0, icpFitness, icpTime);
            
            // tf_Lprior_L0 = myTf(tfm_Lprior_L0).cast<double>();

            // printf(KGRN "Refine the transform: %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f. Fitness: %f. Time: %f\n" RESET,
            //         tf_Lprior_L0.pos.x(), tf_Lprior_L0.pos.y(), tf_Lprior_L0.pos.z(),
            //         tf_Lprior_L0.yaw(), tf_Lprior_L0.pitch(), tf_Lprior_L0.roll(),
            //         icpFitness, icpTime);

            IOAOptions ioaOpt;
            ioaOpt.init_tf = tf_Lprior_L0;
            ioaOpt.max_iterations = ioa_max_iter;
            ioaOpt.show_report = true;
            ioaOpt.text = "T_Lprior_L0_refined_" + std::to_string(ioa_max_iter);
            // ioaOpt.fix_rot = fix_rot;
            // ioaOpt.fix_trans = fix_trans;
            IOASummary ioaSum;
            cm.IterateAssociateOptimize(ioaOpt, ioaSum, localMap, KfCloudinW.back());

            tf_Lprior_L0 = ioaSum.final_tf;

            printf(KGRN "Refined the transform: %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f. Time: %f\n" RESET,
                    tf_Lprior_L0.pos.x(), tf_Lprior_L0.pos.y(), tf_Lprior_L0.pos.z(),
                    tf_Lprior_L0.yaw(), tf_Lprior_L0.pitch(), tf_Lprior_L0.roll(), ioaSum.process_time);
        }
        
        // Create an auto exitting scope
        {
            lock_guard<mutex>lg(relocBufMtx);
            relocBuf.push_back(tf_Lprior_L0);
        }
    }

    void PublishPriorMap(const ros::TimerEvent& event)
    {
        Util::publishCloud(priorMapPub, *priorMap, ros::Time::now(), "map");
    }

    bool GetBoolParam(string param, bool default_value)
    {
        int param_;
        nh_ptr->param(param, param_, default_value == true ? 1 : 0);
        return (param_ == 0 ? false : true);
    }

    bool TimeIsValid(PoseSplineX &traj, double time, double tolerance = 0)
    {
        return traj.minTime() + tolerance < time && time < traj.maxTime() - tolerance;
    }

    void DataHandler(const slict::FeatureCloud::ConstPtr &msg)
    {
        lock_guard<mutex> lock(packet_buf_mtx);
        packet_buf.push_back(msg);
    }

    /**
     * SLAM系统的主数据处理函数
     * 功能描述：处理传感器数据，执行SLAM核心算法流程
     * 主要步骤：
     *   1. 数据检查和提取
     *   2. 传感器初始化
     *   3. IMU传播和点云去畸变
     *   4. 点云-地图关联
     *   5. LIO优化
     *   6. 关键帧管理和闭环检测
     *   7. 结果可视化和窗口滑动
     * @return bool 处理是否成功（实际上是无限循环直到程序退出）
     */
    bool ProcessData()
    {
        while (ros::ok())
        {
            // 时间日志记录器和主循环计时器
            slict::TimeLog tlog; TicToc tt_whileloop;

            // 检查数据超时以退出程序
            static TicToc tt_time_out;

            static double data_time_out = -1;
            // 检查数据超时情况：如果20秒内没有新数据且缓冲区为空，则退出程序
            if ( (data_time_out != -1) && (tt_time_out.Toc()/1000.0 - data_time_out) > 20 && (packet_buf.size() == 0) && autoexit)
            {
                printf(KYEL "Data timeout, Buf: %d. exit!\n" RESET, packet_buf.size());
                SaveTrajLog();  // 保存轨迹日志
                exit(0);        // 退出程序
            }

            /* #region STEP 0: 检查是否有新数据，无数据时循环等待 ---------------------------------------------------------*/
            
            TicToc tt_loop;

            // 如果数据包缓冲区为空，则等待1毫秒后继续循环
            if (packet_buf.empty())
            {
                this_thread::sleep_for(chrono::milliseconds(1));
                continue;
            }

            /* #endregion STEP 0: 检查是否有新数据，无数据时循环等待 ------------------------------------------------------*/

            /* #region STEP 1: 提取数据包 --------------------------------------------------------------*/
            
            tt_preopt.Tic();  // 开始计时预处理优化时间

            TicToc tt_extract; // 提取数据包计时器

            slict::FeatureCloud::ConstPtr packet;
            {
                // 使用互斥锁保护数据包缓冲区，确保线程安全
                lock_guard<mutex> lock(packet_buf_mtx);
                packet = packet_buf.front();  // 获取队首数据包
                packet_buf.pop_front();       // 从队列中移除已处理的数据包
            }

            // 重置超时计时器，记录最后一次接收数据的时间
            data_time_out = tt_time_out.Toc()/1000.0;

            tt_extract.Toc(); // 停止提取数据包的计时

            /* #endregion STEP 1: 提取数据包 -----------------------------------------------------------*/

            /* #region STEP 2: 初始化传感器方向和地图 -------------------------------------------------------*/
            
            TicToc tt_init; // 初始化计时器

            // 检查系统是否已完全初始化
            if (!ALL_INITED)
            {
                InitSensorData(packet); // 使用当前数据包初始化传感器数据
                if (!ALL_INITED)        // 如果初始化未完成，继续下一次循环
                    continue;
            }

            tt_init.Toc(); // 停止初始化计时

            /* #endregion STEP 2: 初始化传感器方向和地图 ----------------------------------------------------*/

            /* #region STEP 3: 将数据插入到缓冲区 -------------------------------------------------------*/
            
            TicToc tt_insert; // 数据插入计时器

            // 扩展时间步骤序列，添加新的时间段
            AddNewTimeStep(SwTimeStep, packet);

            // 复制点云数据到滑动窗口
            SwCloud.push_back(CloudXYZITPtr(new CloudXYZIT())); // 创建新的点云容器
            pcl::fromROSMsg(packet->extracted_cloud, *SwCloud.back()); // 从ROS消息转换点云

            // 对扫描进行下采样处理
            if(leaf_size > 0)
            {
                pcl::UniformSampling<PointXYZIT> downsampler;     // 均匀下采样器
                downsampler.setRadiusSearch(leaf_size);           // 设置下采样半径
                downsampler.setInputCloud(SwCloud.back());        // 设置输入点云
                downsampler.filter(*SwCloud.back());              // 执行下采样过滤
            }

            // 为去畸变后的点云创建容器
            SwCloudDsk.push_back(CloudXYZIPtr(new CloudXYZI()));

            // 为下采样的去畸变点云创建容器
            SwCloudDskDS.push_back(CloudXYZIPtr(new CloudXYZI()));

            // 存储激光雷达因子系数的缓冲区
            SwLidarCoef.push_back(vector<LidarCoef>());

            // 记录每个体素关联数量的缓冲区
            SwDepVsAssoc.push_back(map<int, int>());

            // 将IMU样本添加到最后状态的缓冲区
            AddImuToBuff(SwTimeStep, SwImuBundle, packet, regularize_imu);

            // IMU传播状态序列（每个时间步包含N_SUB_SEG个子段）
            SwPropState.push_back(deque<ImuProp>(N_SUB_SEG));

            // 扩展B样条轨迹
            if (GlobalTraj == nullptr)
            {
                // 首次创建全局轨迹样条
                GlobalTraj = PoseSplinePtr(new PoseSplineX(SPLINE_N, deltaT));
                GlobalTraj->setStartTime(SwTimeStep.front().front().start_time);
                printf("Creating spline of order %d, dt %f s. Time: %f\n", SPLINE_N, deltaT, SwTimeStep.front().front().start_time);
            }
            // 扩展样条节点到当前时间步的结束时间
            GlobalTraj->extendKnotsTo(SwTimeStep.back().back().final_time, SE3d());
            
            tlog.t_insert = tt_insert.Toc(); // 记录数据插入时间

            /* #endregion STEP 3: 将数据插入到缓冲区 ----------------------------------------------------*/

            /* #region STEP 4: 在最后的时间段上执行IMU传播 -------------------------------------------------*/

            TicToc tt_imuprop; // IMU传播计时器

            // 遍历最新时间步中的所有子段，执行IMU传播
            for(int i = 0; i < SwImuBundle.back().size(); i++)
            {
                auto &imuSubSeq = SwImuBundle.back()[i];   // 当前子段的IMU数据序列
                auto &subSegment = SwTimeStep.back()[i];   // 当前子段的时间信息

                // 假设条件：IMU数据不中断（任何子段都应该有IMU数据）
                // printf("Step: %2d / %2d. imuSubSeq.size(): %d\n", i, SwImuBundle.back().size(), imuSubSeq.size());
                ROS_ASSERT(!imuSubSeq.empty());

                // 假设条件：在每个段的开始和结束点都有插值的IMU样本
                ROS_ASSERT_MSG(imuSubSeq.front().t == subSegment.start_time && imuSubSeq.back().t == subSegment.final_time,
                               "IMU Time: %f, %f. Seg. Time: %f, %f\n",
                               imuSubSeq.front().t, imuSubSeq.back().t, subSegment.start_time, subSegment.final_time);

                // 使用初始状态和IMU数据执行传播
                SwPropState.back()[i] = ImuProp(ssQua.back()[i], ssPos.back()[i], ssVel.back()[i],
                                                ssBig.back()[i], ssBia.back()[i], GRAV, imuSubSeq);

                // 提取传播后的终止状态
                sfQua.back()[i] = SwPropState.back()[i].Q.back(); // 终止时刻的四元数
                sfPos.back()[i] = SwPropState.back()[i].P.back(); // 终止时刻的位置
                sfVel.back()[i] = SwPropState.back()[i].V.back(); // 终止时刻的速度

                // 用前一段的终止传播状态初始化下一段的开始状态
                if (i <= SwImuBundle.back().size() - 2)
                {
                    ssQua.back()[i+1] = sfQua.back()[i]; // 下一段开始的四元数
                    ssPos.back()[i+1] = sfPos.back()[i]; // 下一段开始的位置
                    ssVel.back()[i+1] = sfVel.back()[i]; // 下一段开始的速度
                }
            }

            tlog.t_prop.push_back(tt_imuprop.Toc()); // 记录IMU传播时间

            /* #endregion STEP 4: 在最后的时间段上执行IMU传播 ----------------------------------------------*/

            /* #region STEP 5: 初始化样条的扩展部分 --------------------------------------------*/

            TicToc tt_extspline; // 样条扩展计时器
            
            static int last_updated_knot = -1; // 记录最后更新的节点索引

            // 使用传播值初始化样条节点
            int baseKnot = GlobalTraj->computeTIndex(SwPropState.back().front().t[0]).second + 1;
            for(int knot_idx = baseKnot; knot_idx < GlobalTraj->numKnots(); knot_idx++)
            {
                // 通过线性插值进行初始化
                double knot_time = GlobalTraj->getKnotTime(knot_idx);
                
                // 查找传播的位姿
                for(int seg_idx = 0; seg_idx < SwPropState.size(); seg_idx++)
                {
                    for(int subseg_idx = 0; subseg_idx < SwPropState.back().size(); subseg_idx++)
                    {
                        // 如果当前子段的结束时间小于节点时间，继续下一个
                        if(SwPropState.back()[subseg_idx].t.back() < knot_time)
                            continue;

                        // 使用传播状态在节点时间的位姿设置节点值
                        GlobalTraj->setKnot(SwPropState.back()[subseg_idx].getTf(knot_time).getSE3(), knot_idx);
                        break;
                    }
                }

                // 通过复制前一个控制点进行初始化
                if (GlobalTraj->getKnotTime(knot_idx) >= GlobalTraj->maxTime()
                    || GlobalTraj->getKnotTime(knot_idx) >= SwPropState.back().back().t.back())
                {
                    // GlobalTraj->setKnot(SwPropState.back().back().getTf(knot_time, false).getSE3(), knot_idx);
                    GlobalTraj->setKnot(GlobalTraj->getKnot(knot_idx-1), knot_idx); // 复制前一个节点的值
                    continue;
                }
            }
            
            // 在结束段拟合样条以避免高成本
            std::thread threadFitSpline;
            if (fit_spline)
            {
                static bool fit_spline_enabled = false;
                // 当滑动窗口达到要求大小时启用样条拟合
                if (SwTimeStep.size() >= WINDOW_SIZE && !fit_spline_enabled)
                    fit_spline_enabled = true;
                else if(fit_spline_enabled)
                    threadFitSpline = std::thread(&Estimator::FitSpline, this); // 在新线程中执行样条拟合
            }

            tt_extspline.Toc(); // 停止样条扩展计时

            /* #endregion STEP 5: 初始化样条的扩展部分 -----------------------------------------*/

            // 检查滑动窗口是否达到所需长度，未达到则继续循环
            if (SwTimeStep.size() < WINDOW_SIZE)
            {
                printf(KGRN "Buffer size %02d / %02d\n" RESET, SwTimeStep.size(), WINDOW_SIZE);
                continue;
            }
            else
            {
                static bool first_shot = true;
                if (first_shot)
                {
                    first_shot = false;
                    printf(KGRN "Buffer size %02d / %02d. WINDOW SIZE reached.\n" RESET, SwTimeStep.size(), WINDOW_SIZE);
                }
            }

            // 如果ufomap已更新，重置关联状态
            static int last_ufomap_version = ufomap_version;
            static bool first_round = true;
            if (ufomap_version != last_ufomap_version)
            {
                first_round   = true;              // 标记为第一轮处理
                last_ufomap_version = ufomap_version;

                printf(KYEL "UFOMAP RESET.\n" RESET);
            }

            /* #region STEP 6: 对点云进行去畸变处理 ----------------------------------------------------------------*/
            
            string t_deskew;   // 去畸变时间报告字符串
            double tt_deskew = 0; // 总去畸变时间

            // 遍历滑动窗口中的帧，执行去畸变
            for (int i = first_round ? 0 : WINDOW_SIZE - 1; i < WINDOW_SIZE; i++)
            {
                TicToc tt; // 单帧去畸变计时器

                // 根据去畸变方法选择不同的处理方式
                switch (deskew_method[0])
                {
                    case 0:
                        // 使用IMU数据进行去畸变
                        DeskewByImu(SwPropState[i], SwTimeStep[i], SwCloud[i], SwCloudDsk[i], SwCloudDskDS[i], assoc_spacing);
                        break;
                    case 1:
                        // 使用B样条进行去畸变
                        DeskewBySpline(*GlobalTraj, SwTimeStep[i], SwCloud[i], SwCloudDsk[i], SwCloudDskDS[i], assoc_spacing);
                        break;
                    default:
                        break;
                }

                // 记录计时信息
                tt.Toc();
                t_deskew  += myprintf("#%d: %3.1f, ", i, tt.GetLastStop());
                tt_deskew += tt.GetLastStop();
            }

            // 格式化去畸变时间报告
            t_deskew = myprintf("deskew: %3.1f, ", tt_deskew) + t_deskew;

            tlog.t_desk.push_back(tt_deskew); // 记录去畸变时间到日志

            /* #endregion STEP 6: 对点云进行去畸变处理 -------------------------------------------------------------*/

            /* #region STEP 7: 扫描与地图关联 --------------------------------------------------------------*/
            
            string t_assoc;   // 关联时间报告字符串
            double tt_assoc = 0; // 总关联时间

            // 遍历滑动窗口中的帧，执行点云与地图的关联
            for (int i = first_round ? 0 : WINDOW_SIZE - 1; i < WINDOW_SIZE; i++)
            {
                TicToc tt; // 单帧关联计时器

                // 使用互斥锁保护地图数据，确保线程安全
                lock_guard<mutex> lg(map_mtx);
                SwDepVsAssoc[i].clear(); // 清空深度-关联计数映射
                SwLidarCoef[i].clear();  // 清空激光雷达系数缓冲区
                
                // 执行点云与地图的关联，计算观测残差和雅可比
                AssociateCloudWithMap(*activeSurfelMap, activeikdtMap, mytf(sfQua[i].back(), sfPos[i].back()),
                                       SwCloud[i], SwCloudDskDS[i], SwLidarCoef[i], SwDepVsAssoc[i]);

                // 记录计时信息
                tt.Toc();
                t_assoc  += myprintf("#%d: %3.1f, ", i, tt.GetLastStop());
                tt_assoc += tt.GetLastStop();
            }
            
            // 格式化关联时间报告
            t_assoc = myprintf("assoc: %3.1f, ", tt_assoc) + t_assoc;

            tlog.t_assoc.push_back(tt_assoc); // 记录关联时间到日志

            // find_new_node = false; // 调试用代码（已注释）

            /* #endregion STEP 7: 扫描与地图关联 -----------------------------------------------------------*/

            /* #region STEP 8: 激光雷达-惯导融合优化 ----------------------------------------------------------------------*/

            tt_preopt.Toc(); // 结束预优化计时

            static int optNum = 0; optNum++; // 优化计数器
            vector<slict::OptStat> optreport(max_outer_iters); // 优化报告数组
            for (auto &report : optreport)
                report.OptNum = optNum;

            // 更新时间日志检查
            tlog.OptNum = optNum;
            tlog.header.stamp = ros::Time(SwTimeStep.back().back().final_time);

            string printout, lioop_times_report = "", DVAReport; // 输出报告字符串

            // 外层优化循环
            int outer_iter = max_outer_iters;
            while(true)
            {
                // 递减外层迭代计数器
                outer_iter--;

                // 准备当前迭代的报告
                slict::OptStat &report = optreport[outer_iter];

                // 计算各深度层级的下采样率
                makeDVAReport(SwDepVsAssoc, DVA, total_lidar_coef, DVAReport);
                lidar_ds_rate = (max_lidar_factor == -1 ? 1 : max(1, (int)std::floor( (double)total_lidar_coef/max_lidar_factor) ));

                // if(threadFitSpline.joinable())
                //     threadFitSpline.join();

                // 创建局部样条来存储新的节点，将位姿从全局轨迹中分离出来
                PoseSplineX LocalTraj(SPLINE_N, deltaT);
                int swBaseKnot = GlobalTraj->computeTIndex(SwImuBundle[0].front().front().t).second; // 滑动窗口基础节点索引
                int swNextBase = GlobalTraj->computeTIndex(SwImuBundle[1].front().front().t).second; // 下一个基础节点索引

                static map<int, int> prev_knot_x; // 前一次的节点索引映射
                static map<int, int> curr_knot_x; // 当前的节点索引映射

                double swStartTime = GlobalTraj->getKnotTime(swBaseKnot);              // 滑动窗口开始时间
                double swFinalTime = SwTimeStep.back().back().final_time - 1e-3;      // 滑动窗口结束时间（减小偏移避免舍入误差）

                LocalTraj.setStartTime(swStartTime);                    // 设置局部轨迹开始时间
                LocalTraj.extendKnotsTo(swFinalTime, SE3d());           // 扩展局部轨迹节点到结束时间

                // 从全局轨迹复制节点值到局部轨迹
                for(int knot_idx = swBaseKnot; knot_idx < GlobalTraj->numKnots(); knot_idx++)
                {
                    // 检查索引边界，避免越界
                    if ((knot_idx - swBaseKnot) > LocalTraj.numKnots() - 1)
                        continue;
                    LocalTraj.setKnot(GlobalTraj->getKnot(knot_idx), knot_idx - swBaseKnot);
                }

                // 检查节点数量的合理性
                ROS_ASSERT_MSG(LocalTraj.numKnots() <= GlobalTraj->numKnots() - swBaseKnot,
                               "Knot count not matching %d, %d, %d\n",
                               LocalTraj.numKnots(), GlobalTraj->numKnots() - swBaseKnot, swBaseKnot);

                // 为边际化记录节点索引映射
                curr_knot_x.clear();
                for(int knot_idx = 0; knot_idx < LocalTraj.numKnots(); knot_idx++)
                    curr_knot_x[knot_idx + swBaseKnot] = knot_idx;

                TicToc tt_feaSel; // 特征选择计时器
                
                // 选择用于优化的特征
                vector<ImuIdx> imuSelected;        // 选中的IMU因子索引
                vector<lidarFeaIdx> featureSelected; // 选中的激光雷达特征索引
                FactorSelection(LocalTraj, imuSelected, featureSelected);
                
                // 可视化选中的特征
                PublishAssocCloud(featureSelected, SwLidarCoef);
                tt_feaSel.Toc();

                tlog.t_feasel.push_back(tt_feaSel.GetLastStop()); // 记录特征选择时间

                // 执行激光雷达-惯导融合优化
                lioop_times_report = "";
                LIOOptimization(report, lioop_times_report, LocalTraj,
                                prev_knot_x, curr_knot_x, swNextBase, outer_iter,
                                imuSelected, featureSelected, tlog);

                // 将优化后的节点值加载回全局轨迹
                for(int knot_idx = 0; knot_idx < LocalTraj.numKnots(); knot_idx++)
                {
                    GlobalTraj->setKnot(LocalTraj.getKnot(knot_idx), knot_idx + swBaseKnot);
                    last_updated_knot = knot_idx + swBaseKnot;
                }

                /* #region 优化后处理 ------------------------------------------------------------------------*/

                TicToc tt_posproc; // 后处理计时器

                string pstop_times_report = "pp: "; // 后处理时间报告

                // 如果优化快速完成，提前跳出循环
                bool redo_optimization = true;
                if (outer_iter <= 0
                    || (outer_iter <= max_outer_iters - 2
                        && report.JK < report.J0
                        && (report.J0 - report.JK)/report.J0 < dj_thres )
                   )
                    redo_optimization = false;

                // int PROP_THREADS = std::min(WINDOW_SIZE, MAX_THREADS);
                // 重新执行IMU传播

                TicToc tt_prop_; // 传播计时器

                // 使用并行处理重新计算所有滑动窗口帧的IMU传播
                #pragma omp parallel for num_threads(WINDOW_SIZE)
                for (int i = 0; i < WINDOW_SIZE; i++)
                {
                    for(int j = 0; j < SwTimeStep[i].size(); j++)
                    {
                        // 使用优化后的状态重新执行IMU传播
                        SwPropState[i][j] = ImuProp(sfQua[i][j], sfPos[i][j], sfVel[i][j],
                                                    sfBig[i][j], sfBia[i][j], GRAV, SwImuBundle[i][j], -1);

                        // 更新第一帧第一段的起始状态
                        if (i == 0 && j == 0)
                        {
                            ssQua[i][j] = SwPropState[i][j].Q.front();
                            ssPos[i][j] = SwPropState[i][j].P.front();
                            ssVel[i][j] = SwPropState[i][j].V.front();
                        }
                    }
                }

                tlog.t_prop.push_back(tt_prop_.Toc());                         // 记录传播时间
                pstop_times_report += myprintf("prop: %.1f, ", tlog.t_prop.back()); // 添加到报告

                TicToc tt_deskew_; // 重新去畸变计时器

                // 重新执行点云去畸变
                if(lite_redeskew)
                    if (redo_optimization)  // 如果还要再次优化，使用轻量级去畸变方法
                        for (int i = first_round ? 0 : max(0, WINDOW_SIZE - reassociate_steps); i < WINDOW_SIZE; i++)
                            Redeskew(SwPropState[i], SwTimeStep[i], SwCloud[i], SwCloudDskDS[i]);
                    else                    // 如果不再优化，使用完整的去畸变方法
                        for (int i = first_round ? 0 : max(0, WINDOW_SIZE - reassociate_steps); i < WINDOW_SIZE; i++)
                            DeskewByImu(SwPropState[i], SwTimeStep[i], SwCloud[i], SwCloudDsk[i], SwCloudDskDS[i], assoc_spacing);
                else
                    // 始终使用完整的IMU去畸变方法
                    for (int i = first_round ? 0 : max(0, WINDOW_SIZE - reassociate_steps); i < WINDOW_SIZE; i++)
                        DeskewByImu(SwPropState[i], SwTimeStep[i], SwCloud[i], SwCloudDsk[i], SwCloudDskDS[i], assoc_spacing);

                tlog.t_desk.push_back(tt_deskew_.Toc());                        // 记录去畸变时间
                pstop_times_report += myprintf("dsk: %.1f, ", tlog.t_desk.back()); // 添加到报告

                TicToc tt_assoc_; // 重新关联计时器
    
                // 重新执行地图关联
                for (int i = first_round ? 0 : max(0, WINDOW_SIZE - reassociate_steps); i < WINDOW_SIZE; i++)
                {
                    // TicToc tt_assoc; // 调试用计时器（已注释）
                    
                    // 使用互斥锁保护地图数据
                    lock_guard<mutex> lg(map_mtx);
                    SwDepVsAssoc[i].clear(); // 清空深度-关联计数映射
                    SwLidarCoef[i].clear();  // 清空激光雷达系数缓冲区
                    
                    // 重新执行点云与地图的关联
                    AssociateCloudWithMap(*activeSurfelMap, activeikdtMap, mytf(sfQua[i].back(), sfPos[i].back()),
                                           SwCloud[i], SwCloudDskDS[i], SwLidarCoef[i], SwDepVsAssoc[i]);

                    // printf("Assoc Time: %f\n", tt_assoc.Toc()); // 调试输出（已注释）
                }

                tlog.t_assoc.push_back(tt_assoc_.Toc());                          // 记录关联时间
                pstop_times_report += myprintf("assoc: %.1f, ", tlog.t_assoc.back()); // 添加到报告

                tt_posproc.Toc(); // 结束后处理计时

                /* #endregion 优化后处理 ---------------------------------------------------------------------*/

                /* #region Write the report -------------------------------------------------------------------------*/

                // Update the report
                report.header.stamp   = ros::Time(SwTimeStep.back().back().final_time);
                report.OptNumSub      = outer_iter + 1;
                report.keyfrm         = KfCloudPose->size();
                report.margPerc       = margPerc;
                report.fixed_knot_min = first_fixed_knot;
                report.fixed_knot_max = last_fixed_knot;

                report.tpreopt        = tt_preopt.GetLastStop();
                report.tpostopt       = tt_posproc.GetLastStop();
                report.tlp            = tt_loop.Toc();
                
                static double last_tmapping = -1;
                if (last_tmapping != tt_margcloud.GetLastStop())
                {
                    report.tmapimg = tt_margcloud.GetLastStop();
                    last_tmapping  = tt_margcloud.GetLastStop();
                }
                else
                    report.tmapimg = -1;

                Vector3d eul_est = Util::Quat2YPR(Quaternd(report.Qest.w, report.Qest.x, report.Qest.y, report.Qest.z));
                Vector3d eul_imu = Util::Quat2YPR(Quaternd(report.Qest.w, report.Qest.x, report.Qest.y, report.Qest.z));
                Vector3d Vest = Vector3d(report.Vest.x, report.Vest.y, report.Vest.z);
                Vector3d Vimu = Vector3d(report.Vimu.x, report.Vimu.y, report.Vimu.z);

                /* #region */
                printout +=
                    show_report ?
                    myprintf("Op#.Oi#: %04d. %2d /%2d. Itr: %2d / %2d. trun: %.3f. %s. RL: %d\n"
                             "tpo: %4.0f. tfs: %4.0f. tbc: %4.0f. tslv: %4.0f / %4.0f. tpp: %4.0f. tlp: %4.0f. tufm: %4.0f. tlpBa: %4.0f.\n"
                             "Ftr: Ldr: %5d / %5d / %5d. IMU: %5d. Prop: %5d. Vel: %2d. Buf: %2d. Kfr: %d. Marg%%: %6.3f. Kfca: %d. "
                             "Fixed: %d -> %d. "
                             "Map: %d\n"
                             "J0:  %15.3f, Ldr: %9.3f. IMU: %9.3f. Prp: %9.3f. Vel: %9.3f.\n"
                             "JK:  %15.3f, Ldr: %9.3f. IMU: %9.3f. Prp: %9.3f. Vel: %9.3f.\n"
                            //  "BiaG: %7.2f, %7.2f, %7.2f. BiaA: %7.2f, %7.2f, %7.2f. (%7.2f, %7.2f, %7.2f), (%7.2f, %7.2f, %7.2f)\n"
                            //  "Eimu: %7.2f, %7.2f, %7.2f. Pimu: %7.2f, %7.2f, %7.2f. Vimu: %7.2f, %7.2f, %7.2f.\n"
                            //  "Eest: %7.2f, %7.2f, %7.2f. Pest: %7.2f, %7.2f, %7.2f. Vest: %7.2f, %7.2f, %7.2f. Spd: %.3f. Dif: %.3f.\n"
                             "DVA:  %s\n",
                             // Time and iterations
                             report.OptNum, report.OptNumSub, max_outer_iters,
                             report.iters, max_iterations,
                             report.trun,
                             reloc_stat == RELOCALIZED ? KYEL "RELOCALIZED!" RESET : "",
                             relocBuf.size(),
                             report.tpreopt,           // time preparing before lio optimization
                             tt_fitspline.GetLastStop(), // time to fit the spline
                             report.tbuildceres,       // time building the ceres problem before solving
                             report.tslv,              // time solving ceres problem
                             t_slv_budget,             // time left to solve the problem
                             report.tpostopt,          // time for post processing
                             report.tlp,               // time packet was extracted up to now
                             report.tmapimg,           // time of last insertion of data to ufomap
                             tt_loopBA.GetLastStop(),  // time checking loop closure
                             // Sliding window stats
                             report.surfFactors, max_lidar_factor, total_lidar_coef,
                             report.imuFactors, report.propFactors, report.velFactors,
                             report.mfcBuf = packet_buf.size(), report.keyfrm, report.margPerc*100, report.kfcand,
                            //  active_knots.begin()->first, active_knots.rbegin()->first,
                             report.fixed_knot_min, report.fixed_knot_max,
                             use_ufm ? activeSurfelMap->size() : activeikdtMap->size(),
                             // Optimization initial costs
                             report.J0, report.J0Surf, report.J0Imu, report.J0Prop, report.J0Vel,
                             // Optimization final costs
                             report.JK, report.JKSurf, report.JKImu, report.JKProp, report.JKVel,
                             // Bias Estimate
                            //  ssBig.back().back().x(), ssBig.back().back().y(), ssBig.back().back().z(),
                            //  ssBia.back().back().x(), ssBia.back().back().y(), ssBia.back().back().z(),
                            //  BG_BOUND(0), BG_BOUND(1), BG_BOUND(2), BA_BOUND(0), BA_BOUND(1), BA_BOUND(2),
                             // Pose Estimate from propogation
                            //  eul_imu.x(), eul_imu.y(), eul_imu.z(),
                            //  report.Pimu.x, report.Pimu.y, report.Pimu.z,
                            //  report.Vimu.x, report.Vimu.y, report.Vimu.z,
                             // Pose Estimate from Optimization
                            //  eul_est.x(), eul_est.y(), eul_est.z(),
                            //  report.Pest.x, report.Pest.y, report.Pest.z,
                            //  report.Vest.x, report.Vest.y, report.Vest.z,
                            //  Vest.norm(), (Vest - Vimu).norm(),
                             // Report on the assocations at different scales
                             DVAReport.c_str())
                    : "\n";
                /* #endregion */

                // Attach the report from loop closure
                /* #region */
                printout +=
                    myprintf("%sBA# %4d. LoopEn: %d. LastFn: %6.3f. Itr: %3d. tslv: %4.0f. trbm: %4.0f. Ftr: RP: %4d. Lp: %4d.\n"
                             "J:  %6.3f -> %6.3f. rP: %6.3f -> %6.3f. Lp: %6.3f -> %6.3f\n" RESET,
                             // Stats
                             baReport.turn % 2 == 0 ? KBLU : KGRN, baReport.turn, loop_en, lastICPFn,
                             baReport.pgopt_iter, baReport.pgopt_time, baReport.rebuildmap_time,
                             baReport.factor_relpose, baReport.factor_loop,
                             // Costs
                             baReport.J0, baReport.JK,
                             baReport.J0_relpose, baReport.JK_relpose,
                             baReport.J0_loop, baReport.JK_loop);

                // Show the preop times
                string preop_times_report = "";
                if (GetBoolParam("/show_preop_times", false))
                {
                    preop_times_report += "Preop: ";
                    preop_times_report += myprintf("insert:  %3.1f, ", tt_insert.GetLastStop());
                    preop_times_report += myprintf("imuprop: %3.1f, ", tt_imuprop.GetLastStop());
                    preop_times_report += myprintf("extspln: %3.1f, ", tt_extspline.GetLastStop());
                    preop_times_report += t_deskew; 
                    preop_times_report += t_assoc;
                    preop_times_report += myprintf("feaSel: %3.1f, ", tt_feaSel.GetLastStop());
                }
                printout += preop_times_report + pstop_times_report + "\n";
                printout += lioop_times_report + "\n";
                /* #endregion */

                // Publish the optimization results
                static ros::Publisher opt_stat_pub = nh_ptr->advertise<slict::OptStat>("/opt_stat", 1);
                opt_stat_pub.publish(report);

                /* #endregion Write the report ----------------------------------------------------------------------*/

                if(!redo_optimization)
                {
                    prev_knot_x = curr_knot_x;
                    break;
                }
            }

            first_round = false;

            /* #endregion STEP 8: LIO optimizaton -------------------------------------------------------------------*/

            /* #region STEP 9: 关键帧招募 ---------------------------------------------------------------------*/
            
            NominateKeyframe(); // 根据当前状态决定是否添加新的关键帧

            /* #endregion STEP 9: 关键帧招募 ------------------------------------------------------------------*/

            /* #region STEP 10: 闭环检测和束调整 ------------------------------------------------------------------*/

            tt_loopBA.Tic(); // 闭环检测和束调整计时器

            // 如果启用闭环检测
            if (loop_en)
            {
                DetectLoop();                // 检测闭环约束
                BundleAdjustment(baReport);  // 执行全局束调整优化
            }

            tt_loopBA.Toc(); // 结束闭环检测和束调整计时

            /* #endregion STEP 10: 闭环检测和束调整 ---------------------------------------------------------------*/

            /* #region STEP 11: 报告输出和可视化 ----------------------------------------------------------------*/ 
            
            // 输出优化报告摘要
            if (show_report)
                cout << printout;

            // 多线程可视化（已注释，改为同步执行）
            // std::thread vizSwTrajThread(&Estimator::VisualizeSwTraj, this);            
            // std::thread vizSwLoopThread(&Estimator::VisualizeLoop, this);
            
            // vizSwTrajThread.join();
            // vizSwLoopThread.join();

            VisualizeSwTraj(); // 可视化滑动窗口轨迹
            VisualizeLoop();   // 可视化闭环约束

            /* #endregion STEP 11: Report and Vizualize -------------------------------------------------------------*/ 

            /* #region STEP 12: 滑动窗口前移 ----------------------------------------------------------------*/

            // 将滑动窗口向前滑动，移除最旧的数据
            SlideWindowForward();

            /* #endregion STEP 12: 滑动窗口前移 -------------------------------------------------------------*/

            /* #region STEP 13: 转换到先验地图坐标系 ---------------------------------------------*/

            // 模拟重定位行为：如果使用先验地图且尚未重定位但有重定位缓冲区数据
            if (use_prior_map && reloc_stat != RELOCALIZED && relocBuf.size() != 0)
            {
                // 提取重定位位姿
                {
                    lock_guard<mutex>lg(relocBufMtx);  // 保护重定位缓冲区
                    tf_Lprior_L0 = relocBuf.back();    // 获取最新的重定位变换
                }

                // 将所有状态转换到新坐标系
                Quaternd q_Lprior_L0 = tf_Lprior_L0.rot;  // 重定位旋转
                Vector3d p_Lprior_L0 = tf_Lprior_L0.pos;  // 重定位平移

                // 将轨迹转换到先验地图坐标系
                for(int knot_idx = 0; knot_idx < GlobalTraj->numKnots(); knot_idx++)
                    GlobalTraj->setKnot(tf_Lprior_L0.getSE3()*GlobalTraj->getKnot(knot_idx), knot_idx);

                // 将所有IMU位姿转换到先验地图坐标系
                for(int i = 0; i < SwPropState.size(); i++)
                {
                    for(int j = 0; j < SwPropState[i].size(); j++)
                    {
                        for (int k = 0; k < SwPropState[i][j].size(); k++)
                        {
                            // 旋转变换：q' = q_transform * q_original
                            SwPropState[i][j].Q[k] = q_Lprior_L0*SwPropState[i][j].Q[k];
                            // 位置变换：p' = q_transform * p_original + t_transform
                            SwPropState[i][j].P[k] = q_Lprior_L0*SwPropState[i][j].P[k] + p_Lprior_L0;
                            // 速度变换：v' = q_transform * v_original
                            SwPropState[i][j].V[k] = q_Lprior_L0*SwPropState[i][j].V[k];
                        }
                    }
                }

                // 转换起始和终止状态到先验地图坐标系
                for(int i = 0; i < ssQua.size(); i++)
                {
                    for(int j = 0; j < ssQua[i].size(); j++)
                    {
                        // 起始状态转换
                        ssQua[i][j] = q_Lprior_L0*ssQua[i][j];
                        ssPos[i][j] = q_Lprior_L0*ssPos[i][j] + p_Lprior_L0;
                        ssVel[i][j] = q_Lprior_L0*ssVel[i][j];

                        // 终止状态转换
                        sfQua[i][j] = q_Lprior_L0*sfQua[i][j];
                        sfPos[i][j] = q_Lprior_L0*sfPos[i][j] + p_Lprior_L0;
                        sfVel[i][j] = q_Lprior_L0*sfVel[i][j];
                    }
                }

                // 转换关键帧位姿到先验地图坐标系
                pcl::transformPointCloud(*KfCloudPose, *KfCloudPose, tf_Lprior_L0.cast<float>().tfMat());
                
                // 清除之前步骤的关联系数
                for(int i = 0; i < SwLidarCoef.size(); i++)
                {
                    SwLidarCoef[i].clear();    // 清空激光雷达系数
                    SwDepVsAssoc[i].clear();   // 清空深度-关联映射
                }
                
                // 更新先验因子的先验值
                mySolver->RelocalizePrior(tf_Lprior_L0.getSE3());

                // 切换到先验地图
                {
                    lock_guard<mutex> lg(map_mtx);  // 保护地图数据
                    
                    if(use_ufm)
                        activeSurfelMap = priorSurfelMapPtr;  // 切换到先验surfel地图
                    else
                        activeikdtMap = priorikdtMapPtr;      // 切换到先验ikd-tree地图

                    ufomap_version++;  // 增加地图版本号，触发重新关联
                }

                // 转换之前的关键帧点云到先验地图坐标系
                #pragma omp parallel for num_threads(MAX_THREADS)
                for(int i = 0; i < KfCloudPose->size(); i++)
                    pcl::transformPointCloud(*KfCloudinW[i], *KfCloudinW[i], tf_Lprior_L0.cast<float>().tfMat());

                // 清空全局地图并准备重新构建
                {
                    lock_guard<mutex> lock(global_map_mtx);  // 保护全局地图
                    globalMap->clear();
                }

                // 记录重定位信息到日志文件
                Matrix4d tfMat_L0_Lprior = tf_Lprior_L0.inverse().tfMat();  // L0到Lprior的变换矩阵
                Matrix4d tfMat_Lprior_L0 = tf_Lprior_L0.tfMat();            // Lprior到L0的变换矩阵

                // 保存从L0到先验地图的变换矩阵
                ofstream prior_fwtf_file;
                prior_fwtf_file.open((log_dir + string("/tf_L0_Lprior.txt")).c_str());
                prior_fwtf_file << std::fixed << std::setprecision(9);
                prior_fwtf_file << tfMat_L0_Lprior;
                prior_fwtf_file.close();

                // 保存从先验地图到L0的变换矩阵
                ofstream prior_rvtf_file;
                prior_rvtf_file.open((log_dir + string("/tf_Lprior_L0.txt")).c_str());
                prior_rvtf_file << std::fixed << std::setprecision(9);
                prior_rvtf_file << tfMat_Lprior_L0;
                prior_rvtf_file.close();

                // 保存重定位详细信息到CSV文件
                ofstream reloc_info_file;
                reloc_info_file.open((log_dir + string("/reloc_info.csv")).c_str());
                reloc_info_file << std::fixed << std::setprecision(9);
                reloc_info_file << "Time, "
                                << "TF_Lprior_L0.x, TF_Lprior_L0.y, TF_Lprior_L0.z, "
                                << "TF_Lprior_L0.qx, TF_Lprior_L0.qy, TF_Lprior_L0.qz, TF_Lprior_L0.qw, "
                                << "TF_Lprior_L0.yaw, TF_Lprior_L0.pitch, TF_Lprior_L0.roll"  << endl;
                reloc_info_file << SwTimeStep.back().back().final_time << ", "  // 时间戳
                                << tf_Lprior_L0.pos.x() << ", " << tf_Lprior_L0.pos.y() << ", " << tf_Lprior_L0.pos.z() << ", "      // 位置
                                << tf_Lprior_L0.rot.x() << ", " << tf_Lprior_L0.rot.y() << ", " << tf_Lprior_L0.rot.z() << ", " << tf_Lprior_L0.rot.w() << ", " // 四元数
                                << tf_Lprior_L0.yaw()   << ", " << tf_Lprior_L0.pitch() << ", " << tf_Lprior_L0.roll()  << endl;    // 欧拉角
                reloc_info_file.close();

                // 更改参考坐标系
                current_ref_frame = "map";

                reloc_stat = RELOCALIZED; // 标记为已重定位
            }

            /* #endregion STEP 13: 转换到先验地图坐标系 ------------------------------------------*/

            // 发布主循环时间日志
            tlog.t_loop = tt_whileloop.Toc();
            static ros::Publisher tlog_pub = nh_ptr->advertise<slict::TimeLog>("/time_log", 100);
            tlog_pub.publish(tlog);
        }
    } // ProcessData() 函数结束

    void PublishAssocCloud(vector<lidarFeaIdx> &featureSelected, deque<vector<LidarCoef>> &SwLidarCoef)
    {
        static CloudXYZIPtr assocCloud(new CloudXYZI());
        assocCloud->resize(featureSelected.size());

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < featureSelected.size(); i++)
        {
            LidarCoef &coef = SwLidarCoef[featureSelected[i].wdidx][featureSelected[i].pointidx];
            assocCloud->points[i].x = coef.finW(0);
            assocCloud->points[i].y = coef.finW(1);
            assocCloud->points[i].z = coef.finW(2);
            assocCloud->points[i].intensity = featureSelected[i].wdidx;
        }

        static ros::Publisher assoc_cloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/assoc_cloud", 100);
        Util::publishCloud(assoc_cloud_pub, *assocCloud, ros::Time(SwTimeStep.back().back().final_time), current_ref_frame);
    }

    /**
     * 传感器数据初始化函数
     * 功能描述：初始化IMU和激光雷达传感器，计算IMU偏置和初始姿态
     * @param packet 输入的特征点云数据包，包含IMU数据和激光雷达数据
     * 主要步骤：
     *   1. IMU初始化：计算陀螺仪偏置、加速度计标定、初始姿态估计
     *   2. 激光雷达初始化：处理第一帧点云、创建初始关键帧
     *   3. 检查初始化完成状态
     */
    void InitSensorData(slict::FeatureCloud::ConstPtr &packet)
    {
        static bool IMU_INITED = false;    // IMU初始化状态标志
        static bool LIDAR_INITED = false;  // 激光雷达初始化状态标志

        // ============= IMU传感器初始化 =============
        if (!IMU_INITED)
        {
            const vector<sensor_msgs::Imu> &imu_bundle = packet->imu_msgs; // 获取IMU数据束

            // 静态缓冲区，用于累积IMU数据进行统计计算
            static vector<Vector3d> gyr_buf;  // 陀螺仪数据缓冲区
            static vector<Vector3d> acc_buf;  // 加速度计数据缓冲区
            static double first_imu_time = imu_bundle.front().header.stamp.toSec(); // 第一个IMU数据的时间戳

            // 将IMU样本推入缓冲区进行累积
            for (auto imu_sample : imu_bundle)
            {
                // 只处理序列号为0的IMU数据（可能是数据筛选条件）
                if (imu_sample.header.seq == 0)
                {
                    // 提取陀螺仪角速度数据
                    gyr_buf.push_back(Vector3d(imu_sample.angular_velocity.x,
                                               imu_sample.angular_velocity.y,
                                               imu_sample.angular_velocity.z));
                    // 提取加速度计线性加速度数据
                    acc_buf.push_back(Vector3d(imu_sample.linear_acceleration.x,
                                               imu_sample.linear_acceleration.y,
                                               imu_sample.linear_acceleration.z));
                }
            }

            // 当累积足够的IMU数据且时间间隔达到初始化要求时，进行IMU参数计算
            if (!gyr_buf.empty() &&
                fabs(imu_bundle.front().header.stamp.toSec() - first_imu_time) > imu_init_time)
            {
                // ========== 计算陀螺仪偏置 ==========
                Vector3d gyr_avr(0, 0, 0);  // 陀螺仪平均值，用于估计偏置
                for (auto gyr_sample : gyr_buf)
                    gyr_avr += gyr_sample;   // 累加所有陀螺仪测量值

                gyr_avr /= gyr_buf.size();   // 计算平均值作为陀螺仪偏置估计

                // ========== 计算初始姿态 ==========
                Vector3d acc_avr(0, 0, 0);  // 加速度计平均值，用于估计重力方向
                for (auto acc_sample : acc_buf)
                    acc_avr += acc_sample;   // 累加所有加速度计测量值

                acc_avr /= acc_buf.size();   // 计算平均值

                // 计算加速度计标定系数（理论重力与测量重力的比值）
                ACC_SCALE = GRAV.norm()/acc_avr.norm();
                
                // 根据重力方向估计初始姿态四元数
                Quaternd q_init(Util::grav2Rot(acc_avr));
                Vector3d ypr = Util::Quat2YPR(q_init);  // 转换为欧拉角用于显示

                // 输出初始化结果信息
                printf("Gyro Bias: %.3f, %.3f, %.3f. Samples: %d. %d\n",
                        gyr_avr(0), gyr_avr(1), gyr_avr(2), gyr_buf.size(), acc_buf.size());
                printf("Init YPR:  %.3f, %.3f, %.3f.\n", ypr(0), ypr(1), ypr(2));

                // ========== 初始化滑动窗口状态变量 ==========
                // 初始化起始和终止姿态四元数（所有窗口和子段都使用相同的初始姿态）
                ssQua = sfQua = deque<deque<Quaternd>>(WINDOW_SIZE, deque<Quaternd>(N_SUB_SEG, q_init));
                // 初始化陀螺仪偏置（使用计算出的偏置值）
                ssBig = sfBig = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, gyr_avr));
                // 初始化加速度计偏置（设为零向量）
                ssBia = sfBia = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, Vector3d(0, 0, 0)));

                IMU_INITED = true;  // 标记IMU初始化完成
            }
        }
    
        // ============= 激光雷达传感器初始化 =============
        if (!LIDAR_INITED)
        {
            // 静态点云容器，用于存储第一帧激光雷达数据
            static CloudXYZITPtr kfCloud0_(new CloudXYZIT());
            pcl::fromROSMsg(packet->extracted_cloud, *kfCloud0_);  // 从ROS消息转换为PCL点云

            // 只有IMU初始化完成后才能进行激光雷达初始化（需要初始姿态信息）
            if(IMU_INITED)
            {   
                // ========== 点云下采样处理 ==========
                pcl::UniformSampling<PointXYZIT> downsampler;  // 创建均匀下采样器
                downsampler.setRadiusSearch(leaf_size);        // 设置下采样半径
                downsampler.setInputCloud(kfCloud0_);          // 设置输入点云
                downsampler.filter(*kfCloud0_);                // 执行下采样过滤

                // ========== 时间窗口裁剪 ==========
                // 只保留最近0.09秒的点云数据，减少计算负担并保证数据时效性
                printf("PointTime: %f -> %f\n", kfCloud0_->points.front().t, kfCloud0_->points.back().t);
                CloudXYZIPtr kfCloud0(new CloudXYZI());  // 创建裁剪后的点云容器（不含时间戳）
                for(PointXYZIT &p : kfCloud0_->points)
                {
                    // 只保留时间戳在最后0.09秒内的点
                    if(p.t > kfCloud0_->points.back().t - 0.09)
                    {
                        PointXYZI pnew;  // 创建新的点（去掉时间戳）
                        pnew.x = p.x; pnew.y = p.y; pnew.z = p.z; pnew.intensity = p.intensity;
                        kfCloud0->push_back(pnew);   
                    }
                }

                // ========== 坐标变换到世界坐标系 ==========
                CloudXYZIPtr kfCloud0InW(new CloudXYZI());  // 世界坐标系下的关键帧点云
                // 使用IMU初始化得到的姿态将点云从传感器坐标系变换到世界坐标系
                pcl::transformPointCloud(*kfCloud0, *kfCloud0InW, Vector3d(0, 0, 0), sfQua[0].back());

                // ========== 创建初始关键帧 ==========
                // 将处理后的点云作为第一个关键帧添加到系统中
                AdmitKeyframe(packet->header.stamp.toSec(),   // 时间戳
                             sfQua[0].back(),                 // 姿态四元数
                             Vector3d(0, 0, 0),               // 位置（初始设为原点）
                             kfCloud0,                        // 传感器坐标系点云
                             kfCloud0InW);                    // 世界坐标系点云

                // ========== 保存关键帧位姿信息 ==========
                // 写入PCD文件用于快速可视化和调试
                PCDWriter writer; 
                writer.writeASCII<PointPose>(log_dir + "/KfCloudPose.pcd", *KfCloudPose, 18);

                LIDAR_INITED = true;  // 标记激光雷达初始化完成
            }
        }

        // ========== 检查初始化完成状态 ==========
        // 只有当IMU和激光雷达都初始化完成后，整个传感器系统才算初始化完成
        if (IMU_INITED && LIDAR_INITED)
            ALL_INITED = true;
    } // InitSensorData() 函数结束

    /**
     * 添加新时间步骤函数
     * 功能描述：向滑动窗口时间步骤队列添加新的时间段，并将其分割为多个子段
     * @param timeStepDeque 时间步骤双端队列（二维结构：每个元素包含多个时间子段）
     * @param packet 输入的特征点云数据包，包含扫描开始和结束时间
     * 主要步骤：
     *   1. 创建新的时间子段序列容器
     *   2. 根据当前时间步骤状态计算时间参数
     *   3. 将时间段均匀分割为N_SUB_SEG个子段
     * 设计理念：为支持B样条轨迹表示，需要将每个激光雷达扫描时间段细分
     */
    void AddNewTimeStep(deque<deque<TimeSegment>> &timeStepDeque, slict::FeatureCloud::ConstPtr &packet)
    {
        // ========== 步骤1: 添加新的时间子段序列容器 ==========
        timeStepDeque.push_back(deque<TimeSegment>()); // 为新的时间步骤创建子段容器

        // ========== 步骤2: 计算时间段参数 ==========
        double start_time, final_time, sub_timestep;
        
        // 根据队列中的时间步骤数量决定时间计算方式
        if (timeStepDeque.size() == 1)
        {
            // 情况1：第一个时间步骤 - 直接使用数据包中的扫描时间
            start_time = packet->scanStartTime;  // 使用扫描开始时间
            final_time = packet->scanEndTime;    // 使用扫描结束时间
            sub_timestep = (final_time - start_time)/N_SUB_SEG; // 计算子时间段长度
        }
        else
        {
            // 情况2：后续时间步骤 - 从前一个时间步骤的结束时间开始
            start_time = timeStepDeque.rbegin()[1].back().final_time; // 获取前一个时间步骤最后子段的结束时间
            final_time = packet->scanEndTime;                         // 使用当前扫描的结束时间
            sub_timestep = (final_time - start_time)/N_SUB_SEG;       // 计算子时间段长度
        }

        // ========== 步骤3: 创建时间子段序列 ==========
        // 将整个时间段均匀分割为N_SUB_SEG个子段，支持B样条的分段表示
        for(int i = 0; i < N_SUB_SEG; i++)
            timeStepDeque.back().push_back(TimeSegment(start_time + i*sub_timestep,      // 子段开始时间
                                                       start_time + (i+1)*sub_timestep)); // 子段结束时间
    } // AddNewTimeStep() 函数结束

    /**
     * 将IMU数据添加到缓冲区函数
     * 功能描述：从数据包中提取IMU数据，进行时间同步和插值处理，然后分配到对应的时间子段中
     * @param timeStepDeque 时间步骤双端队列，定义了时间分段结构
     * @param imuBundleDeque IMU数据束双端队列，存储每个时间步骤的IMU子序列
     * @param packet 输入的特征点云数据包，包含IMU测量数据
     * @param regularize_imu 是否对IMU数据进行规范化处理的标志
     * 主要步骤：
     *   1. 扩展IMU缓冲区结构
     *   2. 提取和规范化IMU数据
     *   3. 处理时间连续性（边界条件）
     *   4. 分配IMU数据到各个时间子段并进行参数化
     */
    void AddImuToBuff(deque<deque<TimeSegment>> &timeStepDeque, deque<deque<ImuSequence>> &imuBundleDeque,
                      slict::FeatureCloud::ConstPtr &packet, bool regularize_imu)
    {
        // ========== 步骤1: 扩展IMU双端队列结构 ==========
        // 为新的时间步骤创建N_SUB_SEG个IMU子序列容器
        imuBundleDeque.push_back(deque<ImuSequence>(N_SUB_SEG));

        // ========== 步骤2: 提取和规范化IMU数据 ==========
        ImuSequence newImuSequence;  // 新的IMU序列容器
        // 从数据包中提取IMU数据，此阶段只选择主要的IMU数据
        ExtractImuData(newImuSequence, packet, regularize_imu);

        // ========== 步骤3: 处理时间连续性 ==========
        // 为确保IMU数据在时间边界处的连续性，需要添加边界样本
        if(timeStepDeque.size() == 1)
        {
            // 情况1：第一个时间步骤 - 复制第一个样本以保证连续性
            newImuSequence.push_front(newImuSequence.front());  // 复制首个IMU样本
            // 将复制的样本时间戳设置为时间段的开始时间
            newImuSequence.front().t = timeStepDeque.front().front().start_time;
        }
        else
            // 情况2：后续时间步骤 - 借用前一个时间间隔的最后一个样本来保证连续性
            newImuSequence.push_front(imuBundleDeque.rbegin()[1].back().back());

        // ========== 步骤4: 将IMU数据分配到各个时间子段 ==========
        // 遍历当前时间步骤的所有子段
        for(int i = 0; i < timeStepDeque.back().size(); i++)
        {
            // 获取当前子段的时间边界
            double start_time = timeStepDeque.back()[i].start_time;  // 子段开始时间
            double final_time = timeStepDeque.back()[i].final_time;  // 子段结束时间
            double dt = final_time - start_time;                     // 子段时间长度
            
            // 从IMU序列中提取当前时间子段内的数据
            imuBundleDeque.back()[i] = newImuSequence.subsequence(start_time, final_time);
            
            // 对当前子段中的每个IMU样本进行参数化处理
            for(int j = 0; j < imuBundleDeque.back()[i].size(); j++)
            {
                imuBundleDeque.back()[i][j].u = start_time;  // u参数：子段开始时间（用于B样条基础）
                // s参数：归一化时间参数 [0,1]，表示在当前子段内的相对位置
                imuBundleDeque.back()[i][j].s = (imuBundleDeque.back()[i][j].t - start_time)/dt;
            }
        }
    } // AddImuToBuff() 函数结束

    void ExtractImuData(ImuSequence &imu_sequence, slict::FeatureCloud::ConstPtr &packet, bool regularize_timestamp)
    {
        // Copy the messages to the deque
        for(auto &imu : packet->imu_msgs)
        {
            imu_sequence.push_back(ImuSample(imu.header.stamp.toSec(),
                                             Vector3d(imu.angular_velocity.x,
                                                      imu.angular_velocity.y,
                                                      imu.angular_velocity.z),
                                             Vector3d(imu.linear_acceleration.x,
                                                      imu.linear_acceleration.y,
                                                      imu.linear_acceleration.z)
                                                      *ACC_SCALE
                                            ));
        }

        if (regularize_timestamp)
        {
            if (imu_sequence.size() <= 2)
                return;

            double t0 = imu_sequence.front().t;
            double tK = imu_sequence.back().t;

            double dt = (tK - t0)/(imu_sequence.size() - 1);
            
            for(int i = 0; i < imu_sequence.size(); i++)
                imu_sequence[i].t = t0 + dt*i;
        }
    }

    // Complete deskew with downsample
    void DeskewByImu(const deque<ImuProp> &imuProp, const deque<TimeSegment> timeSeg,
                     const CloudXYZITPtr &inCloud, CloudXYZIPtr &outCloud, CloudXYZIPtr &outCloudDS,
                     double ds_radius)
    {
        if (!fuse_imu)
        {
            *outCloud = toCloudXYZI(*inCloud);
            
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setRadiusSearch(ds_radius);
            downsampler.setInputCloud(outCloud);
            downsampler.filter(*outCloudDS);

            return;
        }

        int cloud_size = inCloud->size();
        outCloud->resize(cloud_size);
        outCloudDS->resize(cloud_size);

        const double &start_time = timeSeg.front().start_time;
        const double &final_time = timeSeg.back().final_time;
        
        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < cloud_size; i++)
        {
            auto &inPoint = inCloud->points[i];

            double ts = inPoint.t;
            
            // Find the corresponding subsegment
            int seg_idx = -1;
            for(int j = 0; j < timeSeg.size(); j++)
            {
                if(timeSeg[j].start_time <= ts && ts <= timeSeg[j].final_time)
                {
                    seg_idx = j;
                    break;
                }
                else if (timeSeg[j].start_time - 1.0e-6 <= ts && ts < timeSeg[j].start_time)
                {
                    ts = timeSeg[j].start_time;
                    inPoint.t = start_time;
                    seg_idx = j;
                    break;
                }
                else if (timeSeg[j].final_time < ts && ts <= timeSeg[j].final_time - 1.0e-6)
                {
                    ts = timeSeg[j].final_time;
                    inPoint.t = final_time;
                    seg_idx = j;
                    break;
                }
            }

            if(seg_idx == -1)
            {
                printf(KYEL "Point time %f not in segment: [%f, %f]. Discarding\n" RESET, ts, start_time, final_time);
                outCloud->points[i].x = 0; outCloud->points[i].y = 0; outCloud->points[i].z = 0;
                outCloud->points[i].intensity = 0;
                // outCloud->points[i].t = -1; // Mark this point as invalid
                continue;
            }

            // Transform all points to the end of the scan
            myTf T_Bk_Bs = imuProp.back().getBackTf().inverse()*imuProp[seg_idx].getTf(ts);

            Vector3d point_at_end_time = T_Bk_Bs.rot * Vector3d(inPoint.x, inPoint.y, inPoint.z) + T_Bk_Bs.pos;

            outCloud->points[i].x = point_at_end_time.x();
            outCloud->points[i].y = point_at_end_time.y();
            outCloud->points[i].z = point_at_end_time.z();
            outCloud->points[i].intensity = inPoint.intensity;
            // outCloud->points[i].t = inPoint.t;

            outCloudDS->points[i] = outCloud->points[i];
            outCloudDS->points[i].intensity = i;
        }

        // Downsample the pointcloud
        static int step_time;
        static int step_scale;
        // Reset the scale if the time has elapsed
        if (step_time == -1 || timeSeg.front().start_time - step_time > 5.0)
        {
            step_time = timeSeg.front().start_time;
            step_scale = 0;
        }

        if (ds_radius > 0.0)
        {
            int ds_scale = step_scale;
            CloudXYZIPtr tempDSCloud(new CloudXYZI);
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setInputCloud(outCloudDS);
            
            while(true)
            {
                double ds_effective_radius = ds_radius/(std::pow(2, ds_scale));

                downsampler.setRadiusSearch(ds_effective_radius);
                downsampler.setInputCloud(outCloudDS);
                downsampler.filter(*tempDSCloud);

                // If downsampled pointcloud has too few points, relax the ds_radius
                if(tempDSCloud->size() >= 2*max_lidar_factor/WINDOW_SIZE
                    || tempDSCloud->size() == outCloudDS->size()
                    || ds_effective_radius < leaf_size)
                {
                    outCloudDS = tempDSCloud;
                    break;
                }
                else
                {
                    printf(KYEL "Effective assoc_spacing: %f. Points: %d -> %d. Too few points. Relaxing assoc_spacing...\n" RESET,
                                 ds_effective_radius, outCloudDS->size(), tempDSCloud->size());
                    ds_scale++;
                    continue;
                }
            }
            
            if (ds_scale != step_scale)
            {
                step_scale = ds_scale;
                step_time = timeSeg.front().start_time;
            }
        }
    }

    // Only deskew the associated set
    void Redeskew(const deque<ImuProp> &imuProp, const deque<TimeSegment> timeSeg,
                  const CloudXYZITPtr  &inCloud, CloudXYZIPtr &outCloudDS)
    {
        const double &start_time = timeSeg.front().start_time;
        const double &final_time = timeSeg.back().final_time;
      
        int cloud_size = outCloudDS->size();
        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < cloud_size; i++)
        {
            int point_idx = (int)(outCloudDS->points[i].intensity);

            PointXYZIT &pointRaw = inCloud->points[point_idx];
            double ts = pointRaw.t;
            if (ts < 0)
                continue;
            
            // Find the corresponding subsegment
            int seg_idx = -1;
            for(int j = 0; j < timeSeg.size(); j++)
            {
                if(timeSeg[j].start_time <= ts && ts <= timeSeg[j].final_time)
                {
                    seg_idx = j;
                    break;
                }
                else if (timeSeg[j].start_time - 1.0e-6 <= ts && ts < timeSeg[j].start_time)
                {
                    ts = timeSeg[j].start_time;
                    // coef.t_ = start_time;
                    seg_idx = j;
                    break;
                }
                else if (timeSeg[j].final_time < ts && ts <= timeSeg[j].final_time - 1.0e-6)
                {
                    ts = timeSeg[j].final_time;
                    // coef.t_ = final_time;
                    seg_idx = j;
                    break;
                }
            }

            if(seg_idx == -1)
            {
                printf(KYEL "Point time %f not in segment: [%f, %f]. Discarding\n" RESET, ts, start_time, final_time);
                outCloudDS->points[i].x = 0; outCloudDS->points[i].y = 0; outCloudDS->points[i].z = 0;
                outCloudDS->points[i].intensity = 0;
                // outCloud->points[i].t = -1; // Mark this point as invalid
                continue;
            }

            // Transform all points to the end of the scan
            myTf T_Bk_Bs = imuProp.back().getBackTf().inverse()*imuProp[seg_idx].getTf(ts);

            Vector3d point_at_end_time = T_Bk_Bs.rot * Vector3d(pointRaw.x, pointRaw.y, pointRaw.z) + T_Bk_Bs.pos;

            outCloudDS->points[i].x = point_at_end_time.x();
            outCloudDS->points[i].y = point_at_end_time.y();
            outCloudDS->points[i].z = point_at_end_time.z();
            // outCloud->points[i].intensity = inPoint.intensity;
            // outCloud->points[i].t = inPoint.t;

            // outCloudDS->points[i] = outCloud->points[i];
            // outCloudDS->points[i].intensity = i;
        }
    }

    void DeskewBySpline(PoseSplineX &traj, const deque<TimeSegment> timeSeg,
                        const CloudXYZITPtr &inCloud, CloudXYZIPtr &outCloud, CloudXYZIPtr &outCloudDS,
                        double ds_radius)
    {
        if (!fuse_imu)
        {
            *outCloud = toCloudXYZI(*inCloud);
            
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setRadiusSearch(ds_radius);
            downsampler.setInputCloud(outCloud);
            downsampler.filter(*outCloudDS);

            return;
        }

        int cloud_size = inCloud->size();
        outCloud->resize(cloud_size);
        outCloudDS->resize(cloud_size);

        const double &start_time = timeSeg.front().start_time;
        const double &final_time = timeSeg.back().final_time;

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < cloud_size; i++)
        {
            auto &inPoint = inCloud->points[i];

            double ts = inPoint.t;

            if(!TimeIsValid(traj, ts, 1e-6))
            {
                printf(KYEL "Point time %f not in segment: [%f, %f]. Discarding\n" RESET, ts, start_time, final_time);
                outCloud->points[i].x = 0; outCloud->points[i].y = 0; outCloud->points[i].z = 0;
                outCloud->points[i].intensity = 0;
                // outCloud->points[i].t = -1; // Mark this point as invalid
                continue;
            }

            // Transform all points to the end of the scan
            SE3d pose_Bk_Bs = traj.pose(final_time).inverse()*traj.pose(ts);
            myTf T_Bk_Bs(pose_Bk_Bs.so3().unit_quaternion(), pose_Bk_Bs.translation());

            Vector3d point_at_end_time = T_Bk_Bs.rot * Vector3d(inPoint.x, inPoint.y, inPoint.z) + T_Bk_Bs.pos;

            outCloud->points[i].x = point_at_end_time.x();
            outCloud->points[i].y = point_at_end_time.y();
            outCloud->points[i].z = point_at_end_time.z();
            outCloud->points[i].intensity = inPoint.intensity;
            // outCloud->points[i].t = inPoint.t;

            outCloudDS->points[i] = outCloud->points[i];
            // outCloudDS->points[i].intensity = i;
        }

        if (ds_radius > 0.0)
        {
            int ds_scale = 0;
            CloudXYZIPtr tempDSCloud(new CloudXYZI);
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setInputCloud(outCloudDS);
            
            while(true)
            {
                double ds_effective_radius = ds_radius/(std::pow(2, ds_scale));

                downsampler.setRadiusSearch(ds_effective_radius);
                downsampler.setInputCloud(outCloudDS);
                downsampler.filter(*tempDSCloud);

                // If downsampled pointcloud has too few points, relax the ds_radius
                if(tempDSCloud->size() >= 2*max_lidar_factor/WINDOW_SIZE
                    || tempDSCloud->size() == outCloudDS->size()
                    || ds_effective_radius < leaf_size)
                {
                    outCloudDS = tempDSCloud;
                    break;
                }
                else
                {
                    printf(KYEL "Effective assoc_spacing: %f. Points: %d -> %d. Too few points. Relaxing assoc_spacing...\n" RESET,
                                 ds_effective_radius, tempDSCloud->size(), outCloudDS->size());
                    ds_scale++;
                    continue;
                }
            }
        }
    }

    void FitSpline()
    {
        tt_fitspline.Tic();

        // Create a local spline to store the new knots, isolating the poses from the global trajectory
        PoseSplineX SwTraj(SPLINE_N, deltaT);
        int swBaseKnot = GlobalTraj->computeTIndex(SwTimeStep.front().front().start_time).second;

        double swStartTime = GlobalTraj->getKnotTime(swBaseKnot);
        double swFinalTime = SwTimeStep.back().back().final_time;

        SwTraj.setStartTime(swStartTime);
        SwTraj.extendKnotsTo(swFinalTime, SE3d());

        // Copy the knots value
        for(int knot_idx = swBaseKnot; knot_idx < GlobalTraj->numKnots(); knot_idx++)
            SwTraj.setKnot(GlobalTraj->getKnot(knot_idx), knot_idx - swBaseKnot);

        PoseSplineX &traj = SwTraj;

        // Create and solve the Ceres Problem
        ceres::Problem problem;
        ceres::Solver::Options options;

        // Set up the options
        options.linear_solver_type                = linSolver;
        options.trust_region_strategy_type        = trustRegType;
        options.dense_linear_algebra_library_type = linAlgbLib;
        options.max_num_iterations                = max_iterations;
        options.max_solver_time_in_seconds        = max_solve_time;
        options.num_threads                       = MAX_THREADS;
        options.minimizer_progress_to_stdout      = false;

        ceres::LocalParameterization *local_parameterization = new LieAnalyticLocalParameterization<SO3d>();

        // Number of knots of the spline
        int KNOTS = traj.numKnots();

        // Add the parameter blocks for rotational knots
        for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
            problem.AddParameterBlock(traj.getKnotSO3(knot_idx).data(), 4, local_parameterization);

        // Add the parameter blocks for positional knots
        for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
            problem.AddParameterBlock(traj.getKnotPos(knot_idx).data(), 3);

        // Add the parameters for imu biases
        double *BIAS_G = new double[3];
        double *BIAS_A = new double[3];

        BIAS_G[0] = sfBig.back().back().x(); BIAS_A[0] = sfBia.back().back().x();
        BIAS_G[1] = sfBig.back().back().y(); BIAS_A[1] = sfBia.back().back().y();
        BIAS_G[2] = sfBig.back().back().z(); BIAS_A[2] = sfBia.back().back().z();

        problem.AddParameterBlock(BIAS_G, 3);
        problem.AddParameterBlock(BIAS_A, 3);

        for(int i = 0; i < 3; i++)
        {
            if(BG_BOUND(i) > 0)
            {
                problem.SetParameterLowerBound(BIAS_G, i, -BG_BOUND(i));
                problem.SetParameterUpperBound(BIAS_G, i,  BG_BOUND(i));
            }

            if(BA_BOUND(i) > 0)
            {
                problem.SetParameterLowerBound(BIAS_A, i, -BA_BOUND(i));
                problem.SetParameterUpperBound(BIAS_A, i,  BA_BOUND(i));
            }
        }

        for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
        {
            if (traj.getKnotTime(knot_idx) <= SwTimeStep.rbegin()[1].back().final_time)
            {
                problem.SetParameterBlockConstant(traj.getKnotSO3(knot_idx).data());
                problem.SetParameterBlockConstant(traj.getKnotPos(knot_idx).data());
            }
        }

        // Fit the spline with pose and IMU measurements
        vector<ceres::internal::ResidualBlock *> res_ids_poseprop;
        for(int i = WINDOW_SIZE-1; i < WINDOW_SIZE; i++)
        {
            for(int j = 0; j < SwPropState[i].size(); j++)
            {
                for (int k = 0; k < SwPropState[i][j].size()-1; k++)
                {
                    double sample_time = SwPropState[i][j].t[k];

                    // Continue if sample is out of the window
                    if (!traj.TimeIsValid(sample_time, 1e-6))
                        continue;

                    auto   us = traj.computeTIndex(sample_time);
                    double u  = us.first;
                    int    s  = us.second;

                    // Pose
                    ceres::CostFunction *cost_function
                        = new PoseAnalyticFactor
                                (myTf(SwPropState[i][j].Q[k], SwPropState[i][j].P[k]).getSE3(), POSE_N, POSE_N, SPLINE_N, traj.getDt(), u);

                    // Find the coupled poses
                    vector<double *> factor_param_blocks;
                    for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                        factor_param_blocks.emplace_back(traj.getKnotSO3(knot_idx).data());

                    for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                        factor_param_blocks.emplace_back(traj.getKnotPos(knot_idx).data());

                    auto res_block = problem.AddResidualBlock(cost_function, NULL, factor_param_blocks);
                    res_ids_poseprop.push_back(res_block);
                }
            }
        }

        // Add the IMU
        vector<ceres::internal::ResidualBlock *> res_ids_pimu;
        for(int i = WINDOW_SIZE-1; i < WINDOW_SIZE; i++)
        {
            for(int j = 0; j < N_SUB_SEG; j++)
            {
                for(int k = 1; k < SwImuBundle[i][j].size(); k++)
                {
                    double sample_time = SwImuBundle[i][j][k].t;

                    // Skip if sample time exceeds the bound
                    if (!traj.TimeIsValid(sample_time, 1e-6))
                        continue;

                    auto imuBias = ImuBias(Vector3d(BIAS_G[0], BIAS_G[1], BIAS_G[2]),
                                           Vector3d(BIAS_A[0], BIAS_A[1], BIAS_A[2]));

                    auto   us = traj.computeTIndex(sample_time);
                    double u  = us.first;
                    int    s  = us.second;

                    double gyro_weight = GYR_N;
                    double acce_weight = ACC_N;
                    double bgyr_weight = GYR_W;
                    double bacc_weight = ACC_W;

                    ceres::CostFunction *cost_function =
                        new GyroAcceBiasAnalyticFactor
                            (SwImuBundle[i][j][k], imuBias, GRAV, gyro_weight, acce_weight, bgyr_weight, bacc_weight, SPLINE_N, traj.getDt(), u);

                    // Find the coupled poses
                    vector<double *> factor_param_blocks;
                    for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                        factor_param_blocks.emplace_back(traj.getKnotSO3(knot_idx).data());

                    for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                        factor_param_blocks.emplace_back(traj.getKnotPos(knot_idx).data());

                    // gyro bias
                    factor_param_blocks.emplace_back(BIAS_G);

                    // acce bias
                    factor_param_blocks.emplace_back(BIAS_A);

                    // printf("Creating functor: u: %f, s: %d. sample: %d / %d\n", u, s, sample_idx, pose_gt.size());

                    // cost_function->SetNumResiduals(12);
                    ceres::LossFunction* loss_function = imu_loss_thres < 0 ? NULL : new ceres::CauchyLoss(imu_loss_thres);
                    auto res_block = problem.AddResidualBlock(cost_function, loss_function, factor_param_blocks);
                    res_ids_pimu.push_back(res_block);    
                }
            }
        }

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Copy the new knots back to the global trajectory
        for(int knot_idx = 0; knot_idx < SwTraj.numKnots(); knot_idx++)
            GlobalTraj->setKnot(SwTraj.getKnot(knot_idx), knot_idx + swBaseKnot);
        
        delete BIAS_G; delete BIAS_A;

        tt_fitspline.Toc();    
    }
    
    /**
     * 激光雷达-惯导联合优化函数
     * 功能描述：执行SLAM系统的核心优化过程，融合IMU和激光雷达数据约束，估计B样条轨迹和传感器偏差
     * @param report 输出优化统计报告，包含各类因子数量、代价函数值、迭代次数等信息
     * @param lioop_times_report 输出详细的时间分析报告字符串
     * @param traj B样条轨迹对象，包含待优化的位姿节点参数
     * @param prev_knot_x 前一次优化的节点索引映射
     * @param curr_knot_x 当前优化的节点索引映射
     * @param swNextBase 滑动窗口下一个基准索引
     * @param iter 当前优化迭代次数
     * @param imuSelected 选中的IMU因子索引列表
     * @param featureSelected 选中的激光雷达特征因子索引列表
     * @param tlog 时间日志对象，记录各阶段计算时间
     * 
     * 优化策略：
     *   1. 优先尝试自定义求解器（更快速）
     *   2. 如果失败，回退到Ceres优化器（更稳健）
     *   3. 多种因子类型：IMU、激光雷达、位姿传播、速度约束
     *   4. 时间预算管理，确保实时性能
     */
    void LIOOptimization(slict::OptStat &report, string &lioop_times_report, PoseSplineX &traj,
                         map<int, int> &prev_knot_x, map<int, int> &curr_knot_x, int swNextBase, int iter,
                         vector<ImuIdx> &imuSelected,vector<lidarFeaIdx> &featureSelected, slict::TimeLog &tlog)
    {
        // ========== 步骤1: 创建IMU偏差状态 ==========
        // 从滑动窗口最后一帧的最后一个时间段提取陀螺仪和加速度计偏差
        Vector3d XBIG(sfBig.back().back());  // 陀螺仪偏差（Gyroscope Bias）
        Vector3d XBIA(sfBia.back().back());  // 加速度计偏差（Accelerometer Bias）

        // ========== 步骤2: 自定义求解器尝试 ==========
        // 创建自定义TMN求解器（仅在首次调用时初始化）
        if(mySolver == NULL)
            mySolver = new tmnSolver(nh_ptr);

        string iekf_report = "";  // 迭代扩展卡尔曼滤波器报告
        bool ms_success = false;  // 自定义求解器成功标志

        // 尝试使用自定义最小二乘求解器（更快速的解决方案）
        if (!use_ceres)
            ms_success = mySolver->Solve(traj, XBIG, XBIA, prev_knot_x, curr_knot_x, swNextBase, iter,
                                          SwImuBundle, SwCloudDskDS, SwLidarCoef,
                                          imuSelected, featureSelected, iekf_report, report, tlog);

        // ========== 步骤3: 自定义求解器成功后的参数载入 ==========
        if (ms_success)
        {
            /**
             * 参数载入器内部结构体
             * 功能：将B样条轨迹参数转换为状态变量（位姿、速度、偏差）
             */
            struct Loader
            {
                /**
                 * 从B样条轨迹参数复制到状态变量
                 * @param t 时间戳
                 * @param traj B样条轨迹对象
                 * @param ba 加速度计偏差数组指针
                 * @param bg 陀螺仪偏差数组指针
                 * @param BAMAX 加速度计偏差最大值限制
                 * @param BGMAX 陀螺仪偏差最大值限制
                 * @param p_ 输出位置向量
                 * @param q_ 输出旋转四元数
                 * @param v_ 输出速度向量
                 * @param ba_ 输出加速度计偏差向量
                 * @param bg_ 输出陀螺仪偏差向量
                 */
                void CopyParamToState(double t, PoseSplineX &traj, double *ba, double *bg, Vector3d &BAMAX, Vector3d &BGMAX,
                                      Vector3d &p_, Quaternd &q_, Vector3d &v_, Vector3d &ba_, Vector3d &bg_)
                {
                    // 时间边界检查和修正，确保时间在轨迹有效范围内
                    if (t < traj.minTime() + 1e-06)
                    {
                        // printf("State time is earlier than SW time: %f < %f\n", t, traj.minTime());
                        t = traj.minTime() + 1e-06;  // 调整到最小时间边界
                    }

                    if (t > traj.maxTime() - 1e-06)
                    {
                        // printf("State time is later than SW time: %f > %f\n", t, traj.maxTime());
                        t = traj.maxTime() - 1e-06;  // 调整到最大时间边界
                    }

                    // 从B样条轨迹获取指定时间的位姿
                    SE3d pose = traj.pose(t);

                    // 提取位置和旋转
                    p_ = pose.translation();                    // 3D位置
                    q_ = pose.so3().unit_quaternion();         // 单位四元数旋转

                    // 获取世界坐标系下的平移速度
                    v_ = traj.transVelWorld(t);

                    // 加速度计偏差边界约束处理
                    for (int i = 0; i < 3; i++)
                    {
                        if (fabs(ba[i]) > BAMAX[i])
                        {
                            ba_(i) = ba[i]/fabs(ba[i])*BAMAX[i];  // 超出边界时，保持符号但限制幅值
                            break;
                        }
                        else
                            ba_(i) = ba[i];  // 直接赋值
                    }

                    // 陀螺仪偏差边界约束处理
                    for (int i = 0; i < 3; i++)
                    {
                        if (fabs(bg[i]) > BGMAX[i])
                        {
                            bg_(i) = bg[i]/fabs(bg[i])*BGMAX[i];  // 超出边界时，保持符号但限制幅值
                            break;
                        }
                        else
                            bg_(i) = bg[i];  // 直接赋值
                    }

                    // printf("Bg: %f, %f, %f -> %f, %f, %f\n", bg[0], bg[1], bg[2], bg_.x(), bg_.y(), bg_.z());
                    // printf("Ba: %f, %f, %f -> %f, %f, %f\n", ba[0], ba[1], ba[2], ba_.x(), ba_.y(), ba_.z());
                }

            } loader;  // 创建载入器实例

            // 从优化后的B样条轨迹参数载入到滑动窗口状态变量
            for(int i = 0; i < WINDOW_SIZE; i++)           // 遍历滑动窗口中的每一帧
            {
                for(int j = 0; j < SwTimeStep[i].size(); j++)  // 遍历每帧中的每个时间段
                {
                    // 载入时间段起始时刻的状态
                    double ss_time = SwTimeStep[i][j].start_time;
                    loader.CopyParamToState(ss_time, traj, XBIA.data(), XBIG.data(), BA_BOUND, BG_BOUND,
                                            ssPos[i][j], ssQua[i][j], ssVel[i][j], ssBia[i][j], ssBig[i][j]);    

                    // 载入时间段结束时刻的状态
                    double sf_time = SwTimeStep[i][j].final_time;
                    loader.CopyParamToState(sf_time, traj, XBIA.data(), XBIG.data(), BA_BOUND, BG_BOUND,
                                            sfPos[i][j], sfQua[i][j], sfVel[i][j], sfBia[i][j], sfBig[i][j]);

                    // printf("Vel %f: %.2f, %.2f, %.2f\n", sf_time, sfVel[i][j].x(), sfVel[i][j].y(), sfVel[i][j].z());
                }
            }
        } // 自定义求解器成功分支结束

        // ========== 步骤4: Ceres优化器回退方案 ==========
        if(!ms_success)  // 如果自定义求解器失败，使用Ceres优化器
        {
/* #region */ TicToc tt_buildceres;  // 开始计时：Ceres问题构建

/* #region */ TicToc tt_create;      // 开始计时：创建问题和设置选项

            // 创建并配置Ceres优化问题
            ceres::Problem problem;                // Ceres非线性优化问题对象
            ceres::Solver::Options options;       // 求解器选项配置

            // ========== 4.1: 配置求解器选项 ==========
            options.minimizer_type                    = ceres::TRUST_REGION;    // 信赖域方法
            options.linear_solver_type                = linSolver;              // 线性求解器类型
            options.trust_region_strategy_type        = trustRegType;           // 信赖域策略
            options.dense_linear_algebra_library_type = linAlgbLib;             // 稠密线性代数库
            options.max_num_iterations                = max_iterations;         // 最大迭代次数
            options.max_solver_time_in_seconds        = max_solve_time;         // 最大求解时间
            options.num_threads                       = MAX_THREADS;            // 并行线程数
            options.minimizer_progress_to_stdout      = false;                  // 不显示优化进度
            options.use_nonmonotonic_steps            = true;                   // 允许非单调步骤

            // 创建SO(3)李群局部参数化器（处理旋转约束）
            ceres::LocalParameterization *local_parameterization = new LieAnalyticLocalParameterization<SO3d>();

            // ========== 4.2: 获取B样条节点数量 ==========
            int KNOTS = traj.numKnots();  // B样条轨迹的节点总数

            // ========== 4.3: 添加旋转参数块 ==========
            // 为B样条轨迹的每个旋转节点添加参数块（4维四元数，使用李群参数化）
            for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
                problem.AddParameterBlock(traj.getKnotSO3(knot_idx).data(), 4, local_parameterization);

            // ========== 4.4: 添加位置参数块 ==========
            // 为B样条轨迹的每个位置节点添加参数块（3维欧几里得空间）
            for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
                problem.AddParameterBlock(traj.getKnotPos(knot_idx).data(), 3);

            // ========== 4.5: 添加IMU偏差参数块 ==========
            double *BIAS_G = XBIG.data();  // 陀螺仪偏差参数指针
            double *BIAS_A = XBIA.data();  // 加速度计偏差参数指针

            // BIAS_G[0] = sfBig.back().back().x(); BIAS_A[0] = sfBia.back().back().x();
            // BIAS_G[1] = sfBig.back().back().y(); BIAS_A[1] = sfBia.back().back().y();
            // BIAS_G[2] = sfBig.back().back().z(); BIAS_A[2] = sfBia.back().back().z();

            problem.AddParameterBlock(BIAS_G, 3);  // 3维陀螺仪偏差
            problem.AddParameterBlock(BIAS_A, 3);  // 3维加速度计偏差

            // ========== 4.6: IMU偏差边界约束（已禁用）==========
            // for(int i = 0; i < 3; i++)
            // {
            //     if(BG_BOUND[i] > 0)
            //     {
            //         problem.SetParameterLowerBound(BIAS_G, i, -BG_BOUND[i]);  // 陀螺仪偏差下界
            //         problem.SetParameterUpperBound(BIAS_G, i,  BG_BOUND[i]);  // 陀螺仪偏差上界
            //     }

            //     if(BA_BOUND[i] > 0)
            //     {
            //         problem.SetParameterLowerBound(BIAS_A, i, -BA_BOUND[i]);  // 加速度计偏差下界
            //         problem.SetParameterUpperBound(BIAS_A, i,  BA_BOUND[i]);  // 加速度计偏差上界
            //     }
            // }

            // ========== 4.7: 固定滑动窗口起始节点 ==========
            // 固定前几个和后几个节点以提供观测约束和数值稳定性
            first_fixed_knot = -1;  // 第一个固定节点索引
            last_fixed_knot = -1;   // 最后一个固定节点索引
            
            for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
            {
                if (
                    traj.getKnotTime(knot_idx) <= traj.minTime() + start_fix_span  // 在起始固定时间范围内
                    // || traj.getKnotTime(knot_idx) > traj.getKnotTime(KNOTS-1) - final_fix_span  // 在结束固定时间范围内（已禁用）
                )
                {
                    if(first_fixed_knot == -1)
                        first_fixed_knot = knot_idx;  // 记录第一个固定节点

                    last_fixed_knot = knot_idx;       // 更新最后一个固定节点
                    
                    // 固定该节点的旋转和位置参数（设为常量，不参与优化）
                    problem.SetParameterBlockConstant(traj.getKnotSO3(knot_idx).data());
                    problem.SetParameterBlockConstant(traj.getKnotPos(knot_idx).data());
                }
            }

/* #endregion */ tt_create.Toc();  // 结束计时：问题创建

/* #region */ TicToc tt_addlidar;    // 开始计时：添加激光雷达因子

            // ========== 4.8: 添加激光雷达约束因子 ==========
            // 用于可视化关联点云的容器
            CloudXYZIPtr assocCloud(new CloudXYZI());

            // 激光雷达因子相关变量
            vector<ceres::internal::ResidualBlock *> res_ids_surf;  // 表面因子残差块ID列表
            double cost_surf_init = -1, cost_surf_final = -1;       // 初始和最终表面代价函数值
            
            if(fuse_lidar)  // 如果启用激光雷达融合
            {
                // 创建鲁棒损失函数（Cauchy损失或无损失函数）
                ceres::LossFunction *lidar_loss_function = lidar_loss_thres == -1 ? NULL : new ceres::CauchyLoss(lidar_loss_thres);
                int factor_idx = 0;  // 因子索引计数器

                // 遍历选中的激光雷达特征因子并添加到优化问题
                // for (int i = 0; i < WINDOW_SIZE; i++)    // 原始双重循环（已注释）
                // {
                // #pragma omp parallel for num_threads(MAX_THREADS)  // 并行处理（已注释）
                
                for (int j = 0; j < featureSelected.size(); j++)  // 遍历选中的特征因子
                {   
                    // 提取特征因子的索引信息
                    int  i = featureSelected[j].wdidx;      // 滑动窗口索引
                    int  k = featureSelected[j].pointidx;   // 点云中的点索引
                    int  depth = featureSelected[j].depth;  // 关联深度信息

                    auto &point = SwCloudDskDS[i]->points[k];  // 获取对应的点云点
                    int  point_idx = (int)(point.intensity);   // 原始点云中的索引
                    int  coeff_idx = k;                        // 约束系数索引

                    const LidarCoef &coef = SwLidarCoef[i][coeff_idx];  // 获取激光雷达约束系数
                    // ROS_ASSERT_MSG(coef.t >= 0, "i = %d, k = %d, t = %f", i, k, coef.t);
                    double sample_time = coef.t;  // 样本时间戳
                    // ROS_ASSERT(traj.TimeIsValid(sample_time, 1e-6));
                    
                    // 计算B样条参数：u为归一化时间，s为基节点索引
                    auto   us = traj.computeTIndex(sample_time);
                    double u  = us.first;   // 归一化时间参数 [0,1]
                    int    s  = us.second;  // 基节点索引
                    int base_knot = s;  // 基节点索引
                    vector<double*> factor_param_blocks;  // 因子参数块列表
                    
                    // 添加旋转参数块：涉及SPLINE_N个连续节点的旋转参数
                    for (int knot_idx = base_knot; knot_idx < base_knot + SPLINE_N; knot_idx++)
                        factor_param_blocks.push_back(traj.getKnotSO3(knot_idx).data());
                    
                    // 添加位置参数块：涉及SPLINE_N个连续节点的位置参数
                    for (int knot_idx = base_knot; knot_idx < base_knot + SPLINE_N; knot_idx++)
                        factor_param_blocks.push_back(traj.getKnotPos(knot_idx).data());
                    
                    // 配置关联设置（用于地图关联和重关联）
                    factor_idx++;  // 因子计数递增
                    assocSettings settings(use_ufm, reassoc_rate > 0, reassoc_rate,
                                           surfel_min_point, surfel_min_plnrty, surfel_intsect_rad, dis_to_surfel_max,
                                           lidar_weight, i*100000 + factor_idx);
                    
                    // 创建并添加点到平面解析因子
                    typedef PointToPlaneAnalyticFactor<PredType> p2pFactor;
                    auto res = problem.AddResidualBlock(
                                new p2pFactor(coef.finW,           // 世界坐标系中的点位置
                                              coef.f,              // 到平面的距离
                                              coef.n,              // 平面法向量
                                              coef.plnrty*lidar_weight,  // 平面性加权权重
                                              SPLINE_N, traj.getDt(), u,  // B样条参数
                                              activeSurfelMap, *commonPred, activeikdtMap, settings),  // 地图和关联设置
                                              lidar_loss_function,   // 损失函数
                                              factor_param_blocks);  // 参数块列表
                    res_ids_surf.push_back(res);  // 保存残差块ID用于代价计算
                    
                    // 添加点到可视化点云（用于调试和可视化）
                    PointXYZI pointInW; 
                    pointInW.x = coef.finW.x(); 
                    pointInW.y = coef.finW.y(); 
                    pointInW.z = coef.finW.z();
                    assocCloud->push_back(pointInW);
                    assocCloud->points.back().intensity = i;  // 用强度标记所属窗口索引
                }
                // }
            }

/* #endregion */ tt_addlidar.Toc();  // 结束计时：激光雷达因子添加
        
/* #region */ TicToc tt_addimu;       // 开始计时：IMU因子添加

            // ========== 4.9: 添加IMU惯性测量约束因子 ==========
            // 创建并添加新的预积分因子
            vector<ceres::internal::ResidualBlock *> res_ids_pimu;  // IMU因子残差块ID列表
            double cost_pimu_init = -1, cost_pimu_final = -1;       // 初始和最终IMU代价函数值
            // deque<deque<PreintBase *>> local_preints(WINDOW_SIZE, deque<PreintBase *>(N_SUB_SEG));  // 局部预积分（已注释）
            
            if(fuse_imu)  // 如果启用IMU融合
            {
                // 创建IMU鲁棒损失函数
                ceres::LossFunction* loss_function = imu_loss_thres < 0 ? NULL : new ceres::CauchyLoss(imu_loss_thres);

                // 三层循环遍历滑动窗口中的所有IMU数据
                for(int i = 0; i < WINDOW_SIZE; i++)        // 遍历滑动窗口中的每一帧
                {
                    for(int j = 0; j < N_SUB_SEG; j++)      // 遍历每帧中的每个时间子段
                    {
                        for(int k = 1; k < SwImuBundle[i][j].size(); k++)  // 遍历每个子段中的IMU样本（跳过第0个）
                        {
                            double sample_time = SwImuBundle[i][j][k].t;  // 获取IMU样本时间戳
                            
                            // 检查样本时间是否超出轨迹有效范围
                            if (!traj.TimeIsValid(sample_time, 1e-6))
                                continue;  // 超出范围则跳过

                            // 创建IMU偏差对象（包含陀螺仪和加速度计偏差）
                            auto imuBias = ImuBias(Vector3d(BIAS_G[0], BIAS_G[1], BIAS_G[2]),  // 陀螺仪偏差
                                                Vector3d(BIAS_A[0], BIAS_A[1], BIAS_A[2]));   // 加速度计偏差
        
                            // 计算B样条时间索引
                            auto   us = traj.computeTIndex(sample_time);
                            double u  = us.first;   // 归一化时间参数
                            int    s  = us.second;  // 基节点索引

                            // 设置IMU观测权重（来自噪声协方差）
                            double gyro_weight = GYR_N;  // 陀螺仪观测噪声权重
                            double acce_weight = ACC_N;  // 加速度计观测噪声权重
                            double bgyr_weight = GYR_W;  // 陀螺仪偏差随机游走权重
                            double bacc_weight = ACC_W;  // 加速度计偏差随机游走权重

                            // 创建陀螺仪-加速度计-偏差解析因子
                            ceres::CostFunction *cost_function =
                                new GyroAcceBiasAnalyticFactor
                                    (SwImuBundle[i][j][k],    // IMU测量数据
                                     imuBias,                 // IMU偏差
                                     GRAV,                    // 重力向量
                                     gyro_weight, acce_weight, bgyr_weight, bacc_weight,  // 权重参数
                                     SPLINE_N, traj.getDt(), u);  // B样条参数

                            // 找到与该IMU样本时间相关联的B样条节点参数
                            vector<double *> factor_param_blocks;
                            
                            // 添加旋转参数块：涉及SPLINE_N个连续节点
                            for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                                factor_param_blocks.emplace_back(traj.getKnotSO3(knot_idx).data());

                            // 添加位置参数块：涉及SPLINE_N个连续节点
                            for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                                factor_param_blocks.emplace_back(traj.getKnotPos(knot_idx).data());

                            // 添加陀螺仪偏差参数块
                            factor_param_blocks.emplace_back(BIAS_G);

                            // 添加加速度计偏差参数块
                            factor_param_blocks.emplace_back(BIAS_A);

                            // cost_function->SetNumResiduals(12);  // 残差维度：3(gyro) + 3(acce) + 3(gyro_bias) + 3(acce_bias) = 12
                            
                            // 添加IMU残差块到优化问题
                            auto res_block = problem.AddResidualBlock(cost_function, loss_function, factor_param_blocks);
                            res_ids_pimu.push_back(res_block);  // 保存残差块ID用于代价计算    
                        }
                    }
                }
            }

/* #endregion */ tt_addimu.Toc();     // 结束计时：IMU因子添加

/* #region */ TicToc tt_addpp;         // 开始计时：位姿传播因子添加

            // ========== 4.10: 添加位姿传播约束因子 ==========
            vector<ceres::internal::ResidualBlock *> res_ids_poseprop;  // 位姿传播因子残差块ID列表
            double cost_poseprop_init = -1, cost_poseprop_final = -1;   // 初始和最终位姿传播代价值
            
            if(fuse_poseprop)  // 如果启用位姿传播融合
            {
                // 添加位姿传播约束（来自IMU预积分或其他odometry）
                for(int i = 0; i < WINDOW_SIZE - 1 - reassociate_steps; i++)  // 避免重关联步骤的干扰
                {
                    for(int j = 0; j < SwPropState[i].size(); j++)
                    {
                        for (int k = 0; k < SwPropState[i][j].size()-1; k++)
                        {
                            double sample_time = SwPropState[i][j].t[k];

                            // Continue if sample is in the window
                            if (!traj.TimeIsValid(sample_time, 1e-6) || sample_time > traj.minTime() + 0.1)
                                continue;

                            auto   us = traj.computeTIndex(sample_time);
                            double u  = us.first;
                            int    s  = us.second;

                            // Pose
                            ceres::CostFunction *cost_function
                                = new PoseAnalyticFactor
                                        (myTf(SwPropState[i][j].Q[k], SwPropState[i][j].P[k]).getSE3(), POSE_N, POSE_N, SPLINE_N, traj.getDt(), u);

                            // Find the coupled poses
                            vector<double *> factor_param_blocks;
                            for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                                factor_param_blocks.emplace_back(traj.getKnotSO3(knot_idx).data());

                            for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                                factor_param_blocks.emplace_back(traj.getKnotPos(knot_idx).data());

                            auto res_block = problem.AddResidualBlock(cost_function, NULL, factor_param_blocks);
                            res_ids_poseprop.push_back(res_block);
                        }
                    }
                }
            }

            vector<ceres::internal::ResidualBlock *> res_ids_velprop;
            double cost_velprop_init = -1, cost_velprop_final = -1;
            if(fuse_velprop)
            {
                // Add the velocity
                for(int i = 0; i < 1; i++)
                {
                    for(int j = 0; j < SwPropState[i].size(); j++)
                    {
                        for (int k = 0; k < SwPropState[i][j].size()-1; k++)
                        {
                            double sample_time = SwPropState[i][j].t[k];

                            // Continue if sample is in the window
                            if (sample_time < traj.minTime() + 1.0e-6 || sample_time > traj.maxTime() - 1.0e-6)
                                continue;

                            auto   us = traj.computeTIndex(sample_time);
                            double u  = us.first;
                            int    s  = us.second;

                            double &vel_weight = VEL_N;

                            // Velocity
                            ceres::CostFunction *vel_cost_function
                                = new VelocityAnalyticFactor(SwPropState[i][j].V[k], vel_weight, SPLINE_N, traj.getDt(), u);

                            // Find the coupled poses
                            vector<double *> factor_param_blocks;
                            for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                                factor_param_blocks.emplace_back(traj.getKnotPos(knot_idx).data());

                            auto res_block = problem.AddResidualBlock(vel_cost_function, NULL, factor_param_blocks);
                            res_ids_velprop.push_back(res_block);
                        }
                    }
                }
            }

/* #endregion */ tt_addpp.Toc();    // 结束计时：位姿传播因子添加
            
/* #region */ TicToc tt_init_cost;   // 开始计时：初始代价计算

            // ========== 4.12: 计算优化前的初始代价函数值 ==========
            if(find_factor_cost)  // 如果需要计算因子代价（用于分析和调试）
            {
                Util::ComputeCeresCost(res_ids_surf, cost_surf_init, problem);         // 表面因子代价
                Util::ComputeCeresCost(res_ids_pimu, cost_pimu_init, problem);         // IMU因子代价
                Util::ComputeCeresCost(res_ids_poseprop, cost_poseprop_init, problem); // 位姿传播因子代价
                Util::ComputeCeresCost(res_ids_velprop, cost_velprop_init, problem);   // 速度传播因子代价
            }

/* #endregion */ tt_init_cost.Toc(); // 结束计时：初始代价计算
            
/* #endregion */ tt_buildceres.Toc(); tlog.t_prep.push_back(tt_buildceres.GetLastStop()); // 记录预处理时间

/* #region */ TicToc tt_solve;        // 开始计时：优化求解

            // ========== 4.13: 实时性能预算管理 ==========
            if (ensure_real_time)  // 如果需要确保实时性能
            {
                // 动态调整求解时间预算：总预算95% - 已用时间，最少50ms
                t_slv_budget = max(50.0, sweep_len * 95 - (tt_preopt.GetLastStop() + tt_buildceres.GetLastStop()));
                if (packet_buf.size() > 0)  // 如果有待处理数据包，限制求解时间
                    t_slv_budget = 50.0;
                options.max_solver_time_in_seconds = t_slv_budget/1000.0;
            }
            else
                t_slv_budget = options.max_solver_time_in_seconds*1000;

            // ========== 4.14: 执行Ceres非线性优化求解 ==========
            ceres::Solver::Summary summary;       // 求解摘要
            ceres::Solve(options, &problem, &summary);  // 执行优化求解

/* #endregion */ tt_solve.Toc(); tlog.t_compute.push_back(tt_solve.GetLastStop()); // 记录求解时间

/* #region */ TicToc tt_aftsolve;    // 开始计时：优化后处理

/* #region */ TicToc tt_final_cost;  // 开始计时：最终代价计算
            // ========== 4.15: 计算优化后的最终代价函数值 ==========
            if(find_factor_cost)  // 如果需要计算因子代价（用于分析优化效果）
            {
                Util::ComputeCeresCost(res_ids_surf, cost_surf_final, problem);         // 最终表面因子代价
                Util::ComputeCeresCost(res_ids_pimu, cost_pimu_final, problem);         // 最终IMU因子代价
                Util::ComputeCeresCost(res_ids_poseprop, cost_poseprop_final, problem); // 最终位姿传播因子代价
                Util::ComputeCeresCost(res_ids_velprop, cost_velprop_final, problem);   // 最终速度传播因子代价
            }
/* #endregion */ tt_final_cost.Toc(); // 结束计时：最终代价计算

/* #region  */ TicToc tt_load;        // 开始计时：参数载入状态

            // ========== 4.16: 从优化后的参数载入到状态变量 ==========
            struct Loader
            {
                // void CopyStateToParam(Vector3d &p_, Quaternd &q_, Vector3d &v_,
                //                       Vector3d &ba, Vector3d &bg,
                //                       double *&pose, double *&velo, double *&bias)
                // {
                //     pose[0] = p_.x(); pose[1] = p_.y(); pose[2] = p_.z();
                //     pose[3] = q_.x(); pose[4] = q_.y(); pose[5] = q_.z(); pose[6] = q_.w();

                //     velo[0] = v_.x(); velo[1] = v_.y(); velo[2] = v_.z();
                    
                //     bias[0] = ba.x(); bias[1] = ba.y(); bias[2] = ba.z();
                //     bias[3] = bg.x(); bias[4] = bg.y(); bias[5] = bg.z();
                // }

                /**
                 * 从优化后的B样条轨迹参数复制到状态变量（Ceres版本）
                 * 与自定义求解器版本基本相同，但偏差处理略有不同
                 */
                void CopyParamToState(double t, PoseSplineX &traj, double *&ba, double *&bg, Vector3d &BAMAX, Vector3d &BGMAX,
                                    Vector3d &p_, Quaternd &q_, Vector3d &v_, Vector3d &ba_, Vector3d &bg_)
                {

                    if (t < traj.minTime() + 1e-06)
                    {
                        // printf("State time is earlier than SW time: %f < %f\n", t, traj.minTime());
                        t = traj.minTime() + 1e-06;
                    }

                    if (t > traj.maxTime() - 1e-06)
                    {
                        // printf("State time is later than SW time: %f > %f\n", t, traj.maxTime());
                        t = traj.maxTime() - 1e-06;
                    }

                    SE3d pose = traj.pose(t);

                    p_ = pose.translation();
                    q_ = pose.so3().unit_quaternion();

                    v_ = traj.transVelWorld(t);

                    for (int i = 0; i < 3; i++)
                    {
                        if (fabs(ba[i]) > BAMAX[i])
                        {
                            ba_ = Vector3d(0, 0, 0);
                            break;
                        }
                        else
                            ba_(i) = ba[i];
                    }

                    for (int i = 0; i < 3; i++)
                    {
                        if (fabs(bg[i]) > BGMAX[i])
                        {
                            bg_ = Vector3d(0, 0, 0);
                            break;
                        }
                        else
                            bg_(i) = bg[i];
                    }

                    // printf("Bg: %f, %f, %f -> %f, %f, %f\n", bg[0], bg[1], bg[2], bg_.x(), bg_.y(), bg_.z());
                    // printf("Ba: %f, %f, %f -> %f, %f, %f\n", ba[0], ba[1], ba[2], ba_.x(), ba_.y(), ba_.z());
                }

            } loader;

            // Load values from params to state
            for(int i = 0; i < WINDOW_SIZE; i++)
            {
                for(int j = 0; j < SwTimeStep[i].size(); j++)
                {
                    // Load the state at the start time of each segment
                    double ss_time = SwTimeStep[i][j].start_time;
                    loader.CopyParamToState(ss_time, traj, BIAS_A, BIAS_G, BA_BOUND, BG_BOUND,
                                            ssPos[i][j], ssQua[i][j], ssVel[i][j], ssBia[i][j], ssBig[i][j]);    

                    // Load the state at the final time of each segment
                    double sf_time = SwTimeStep[i][j].final_time;
                    loader.CopyParamToState(sf_time, traj, BIAS_A, BIAS_G, BA_BOUND, BG_BOUND,
                                            sfPos[i][j], sfQua[i][j], sfVel[i][j], sfBia[i][j], sfBig[i][j]);

                    // printf("Vel %f: %.2f, %.2f, %.2f\n", sf_time, sfVel[i][j].x(), sfVel[i][j].y(), sfVel[i][j].z());
                }
            }
            
            // delete BIAS_G; delete BIAS_A;

/* #endregion */ tt_load.Toc();
            
/* #region Load data to the report */ TicToc tt_report;  // 开始计时：报告数据载入

            // ========== 4.17: 填充优化统计报告 ==========
            tlog.ceres_iter = summary.iterations.size();  // 记录Ceres迭代次数

            // 表面/激光雷达因子统计
            report.surfFactors = res_ids_surf.size();      // 表面因子数量
            report.J0Surf = cost_surf_init;                // 初始表面代价
            report.JKSurf = cost_surf_final;               // 最终表面代价
            
            // IMU因子统计
            report.imuFactors = res_ids_pimu.size();       // IMU因子数量
            report.J0Imu = cost_pimu_init;                 // 初始IMU代价
            report.JKImu = cost_pimu_final;                // 最终IMU代价

            // 位姿传播因子统计
            report.propFactors = res_ids_poseprop.size();  // 位姿传播因子数量
            report.J0Prop = cost_poseprop_init;            // 初始位姿传播代价
            report.JKProp = cost_poseprop_final;           // 最终位姿传播代价

            // 速度因子统计
            report.velFactors = res_ids_velprop.size();    // 速度因子数量
            report.J0Vel = cost_velprop_init;              // 初始速度代价
            report.JKVel = cost_velprop_final;             // 最终速度代价

            // 总体优化代价统计
            report.J0 = summary.initial_cost;             // 总初始代价
            report.JK = summary.final_cost;               // 总最终代价
            
            report.Qest.x = sfQua.back().back().x();
            report.Qest.y = sfQua.back().back().y();
            report.Qest.z = sfQua.back().back().z();
            report.Qest.w = sfQua.back().back().w();

            report.Pest.x = sfPos.back().back().x();
            report.Pest.y = sfPos.back().back().y();
            report.Pest.z = sfPos.back().back().z();

            report.Vest.x = sfVel.back().back().x();
            report.Vest.y = sfVel.back().back().y();
            report.Vest.z = sfVel.back().back().z();

            report.Qimu.x = SwPropState.back().back().Q.back().x();
            report.Qimu.y = SwPropState.back().back().Q.back().y();
            report.Qimu.z = SwPropState.back().back().Q.back().z();
            report.Qimu.w = SwPropState.back().back().Q.back().w();

            report.Pimu.x = SwPropState.back().back().P.back().x();
            report.Pimu.y = SwPropState.back().back().P.back().y();
            report.Pimu.z = SwPropState.back().back().P.back().z();

            report.Vimu.x = SwPropState.back().back().V.back().x();
            report.Vimu.y = SwPropState.back().back().V.back().y();
            report.Vimu.z = SwPropState.back().back().V.back().z();

            // Calculate the relative pose to the last keyframe
            PointPose lastKf = KfCloudPose->back();
            myTf tf_W_Blast(lastKf);

            report.lastKfId = (int)(lastKf.intensity);
            myTf tf_Blast_Bcurr = tf_W_Blast.inverse()*myTf(sfQua.back().back(), sfPos.back().back());

            report.Qref.x = tf_Blast_Bcurr.rot.x();
            report.Qref.y = tf_Blast_Bcurr.rot.y();
            report.Qref.z = tf_Blast_Bcurr.rot.z();
            report.Qref.w = tf_Blast_Bcurr.rot.w();
            
            report.Pref.x = tf_Blast_Bcurr.pos.x();
            report.Pref.y = tf_Blast_Bcurr.pos.y();
            report.Pref.z = tf_Blast_Bcurr.pos.z();
            
            report.iters = summary.iterations.size();
            report.tbuildceres = tt_buildceres.GetLastStop();
            report.tslv  = tt_solve.GetLastStop();
            report.trun  = (ros::Time::now() - program_start_time).toSec();

            report.BANum            = baReport.turn;
            report.BAItr            = baReport.pgopt_iter;
            report.BALoopTime       = tt_loopBA.GetLastStop();
            report.BASolveTime      = baReport.pgopt_time;
            report.BARelPoseFactors = baReport.factor_relpose;
            report.BALoopFactors    = baReport.factor_loop;
            report.BAJ0             = baReport.J0;
            report.BAJK             = baReport.JK;
            report.BAJ0RelPose      = baReport.J0_relpose;
            report.BAJKRelPose      = baReport.JK_relpose;
            report.BAJ0Loop         = baReport.J0_loop;
            report.BAJKLoop         = baReport.JK_loop;

/* #endregion Load data to the report */ tt_report.Toc();

// /* #region */ TicToc tt_showassoc;

//             // Publish the assoc cloud
//             static ros::Publisher assoc_cloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/assoc_cloud", 100);
//             Util::publishCloud(assoc_cloud_pub, *assocCloud, ros::Time(SwTimeStep.back().back().final_time), current_ref_frame);

// /* #endregion */ tt_showassoc.Toc();

/* #endregion */ tt_aftsolve.Toc(); tlog.t_update.push_back(tt_aftsolve.GetLastStop());

            lioop_times_report = ""; 
            if(GetBoolParam("/show_lioop_times", false))
            {
                lioop_times_report += "lioop: ";
                lioop_times_report += "crt: "     + myprintf("%3.1f. ", tt_create.GetLastStop());
                lioop_times_report += "addldr: "  + myprintf("%4.1f. ", tt_addlidar.GetLastStop());
                lioop_times_report += "addimu: "  + myprintf("%3.1f. ", tt_addimu.GetLastStop());
                lioop_times_report += "addpp: "   + myprintf("%3.1f. ", tt_addpp.GetLastStop());
                lioop_times_report += "J0: "      + myprintf("%4.1f. ", tt_init_cost.GetLastStop());
                lioop_times_report += "bc: "      + myprintf("%4.1f. ", tt_buildceres.GetLastStop());
                lioop_times_report += "slv: "     + myprintf("%4.1f. ", tt_solve.GetLastStop());
                lioop_times_report += "JK: "      + myprintf("%4.1f. ", tt_final_cost.GetLastStop());
                lioop_times_report += "load: "    + myprintf("%3.1f. ", tt_load.GetLastStop());
                lioop_times_report += "rep: "     + myprintf("%3.1f. ", tt_load.GetLastStop());
                // lioop_times_report += "showasc: " + myprintf("%3.1f. ", tt_showassoc.GetLastStop());
                lioop_times_report += "aftslv: "  + myprintf("%3.1f, ", tt_aftsolve.GetLastStop());
                lioop_times_report += "\n";
            }
        } // Ceres优化器分支结束

        // ========== 步骤5: 合并报告信息 ==========
        lioop_times_report += iekf_report;  // 将IEKF报告合并到LIO优化时间报告中
    } // LIOOptimization() 函数结束

    void NominateKeyframe()
    {
        tt_margcloud.Tic();

        int mid_step = 0;//int(std::floor(WINDOW_SIZE/2.0));

        static double last_kf_time = SwTimeStep[mid_step].back().final_time;

        double kf_cand_time = SwTimeStep[mid_step].back().final_time;

        CloudPosePtr kfTempPose(new CloudPose());
        *kfTempPose = *KfCloudPose;

        static KdTreeFLANN<PointPose> kdTreeKeyFrames;
        kdTreeKeyFrames.setInputCloud(kfTempPose);

        myTf tf_W_Bcand(sfQua[mid_step].back(), sfPos[mid_step].back());
        PointPose kf_cand = tf_W_Bcand.Pose6D(kf_cand_time);

        int knn_nbrkf = min(10, (int)kfTempPose->size());
        vector<int> knn_idx(knn_nbrkf); vector<float> knn_sq_dis(knn_nbrkf);
        kdTreeKeyFrames.nearestKSearch(kf_cand, knn_nbrkf, knn_idx, knn_sq_dis);
        
        bool far_distance = knn_sq_dis.front() > kf_min_dis*kf_min_dis;
        bool far_angle = true;
        for(int i = 0; i < knn_idx.size(); i++)
        {
            int kf_idx = knn_idx[i];

            // Collect the angle difference
            Quaternionf Qa(kfTempPose->points[kf_idx].qw,
                           kfTempPose->points[kf_idx].qx,
                           kfTempPose->points[kf_idx].qy,
                           kfTempPose->points[kf_idx].qz);

            Quaternionf Qb(kf_cand.qw, kf_cand.qx, kf_cand.qy, kf_cand.qz);

            // If the angle is more than 10 degrees, add this to the key pose
            if (fabs(Util::angleDiff(Qa, Qb)) < kf_min_angle)
            {
                far_angle = false;
                break;
            }
        }
        bool kf_timeout = fabs(kf_cand_time - last_kf_time) > 2.0 && (knn_sq_dis.front() > 0.1*0.1);

        bool ikdtree_init = false;
        if(!use_ufm)
        {
            static int init_count = 20;
            if(init_count > 0)
            {
                init_count--;
                ikdtree_init = true;
            }
        }

        if (far_distance || far_angle || kf_timeout || ikdtree_init)
        {
            last_kf_time = kf_cand_time;

            static double leaf_sq = pow(leaf_size, 2);

            IOAOptions ioaOpt;
            IOASummary ioaSum;
            ioaSum.final_tf = tf_W_Bcand;
            CloudXYZIPtr marginalizedCloudInW(new CloudXYZI());

            if(refine_kf)
            {
                CloudXYZIPtr localMap(new CloudXYZI());
                // Merge the neighbour cloud
                for(int i = 0; i < knn_idx.size(); i++)
                {
                    int kf_idx = knn_idx[i];
                    *localMap += *KfCloudinW[kf_idx];
                }

                ioaOpt.init_tf = tf_W_Bcand;
                ioaOpt.max_iterations = ioa_max_iter;
                ioaOpt.show_report = false;
                ioaOpt.text = myprintf("Refine T_L_B(%d)_EST", KfCloudPose->size());

                CloudMatcher cm(0.1, 0.1);
                cm.IterateAssociateOptimize(ioaOpt, ioaSum, localMap, SwCloudDsk[mid_step]);

                printf("KF initial: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f.\n"
                       "KF refined: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f.\n",
                        ioaOpt.init_tf.pos.x(), ioaOpt.init_tf.pos.y(), ioaOpt.init_tf.pos.z(),
                        ioaOpt.init_tf.yaw(),   ioaOpt.init_tf.pitch(),
                        ioaOpt.init_tf.roll(),
                        ioaSum.final_tf.pos.x(),
                        ioaSum.final_tf.pos.y(),
                        ioaSum.final_tf.pos.z(),
                        ioaSum.final_tf.yaw(),
                        ioaSum.final_tf.pitch(),
                        ioaSum.final_tf.roll());

                tf_W_Bcand = ioaSum.final_tf;
            }

            if(reloc_stat != RELOCALIZED)
            {
                pcl::transformPointCloud(*SwCloudDsk[mid_step], *marginalizedCloudInW, tf_W_Bcand.cast<float>().tfMat());
            }
            else if(marginalize_new_points)
            {   
                // Skip a few keyframes
                static int count = 5;
                if (count > 0)
                    count--;
                else
                {
                    int new_nodes = 0;
                    int old_nodes = 0;

                    for(int i = 0; i < SwCloudDskDS[mid_step]->size(); i++)
                    {
                        int point_idx = (int)(SwCloudDskDS[mid_step]->points[i].intensity);
                        int coeff_idx = i;
                        if( SwLidarCoef[mid_step][coeff_idx].marginalized )
                        {
                            LidarCoef &coef = SwLidarCoef[mid_step][coeff_idx];
                            
                            ROS_ASSERT(point_idx == coef.ptIdx);

                            PointXYZI pointInB = SwCloudDsk[mid_step]->points[point_idx];
                            
                            PointXYZI pointInW = Util::transform_point(tf_W_Bcand, pointInB);
                            
                            marginalizedCloudInW->push_back(pointInW);
                        }
                    }
                }
            }

            // CloudXYZIPtr marginalizedCloud(new CloudXYZI());
            int margCount = marginalizedCloudInW->size();

            margPerc = double(margCount)/SwCloudDsk[mid_step]->size();
            AdmitKeyframe(SwTimeStep[mid_step].back().final_time, tf_W_Bcand.rot, tf_W_Bcand.pos,
                          SwCloudDsk[mid_step], marginalizedCloudInW);
        }

        tt_margcloud.Toc();
    }

    /**
     * 关键帧接收处理函数
     * 功能描述：接收新的关键帧数据，更新地图并发布相关话题
     * @param t 关键帧时间戳
     * @param q 关键帧姿态四元数（旋转）
     * @param p 关键帧位置向量（平移）
     * @param cloud 传感器坐标系下的关键帧点云
     * @param marginalizedCloudInW 世界坐标系下的边际化点云
     * 主要步骤：
     *   1. 存储关键帧点云数据（传感器坐标系和世界坐标系）
     *   2. 记录关键帧位姿信息
     *   3. 更新全局地图
     *   4. 发布ROS话题用于可视化和其他节点使用
     */
    void AdmitKeyframe(double t, Quaternd q, Vector3d p, CloudXYZIPtr &cloud, CloudXYZIPtr &marginalizedCloudInW)
    {
        tt_ufoupdate.Tic(); // 开始计时UFO地图更新时间

        // ========== 步骤1: 创建关键帧点云容器 ==========
        KfCloudinB.push_back(CloudXYZIPtr(new CloudXYZI())); // 传感器坐标系（Body frame）下的关键帧点云
        KfCloudinW.push_back(CloudXYZIPtr(new CloudXYZI())); // 世界坐标系（World frame）下的关键帧点云

        // ========== 步骤2: 存储和变换点云数据 ==========
        *KfCloudinB.back() = *cloud; // 复制传感器坐标系下的点云
        // 将点云从传感器坐标系变换到世界坐标系
        pcl::transformPointCloud(*KfCloudinB.back(), *KfCloudinW.back(), p, q);

        // ========== 步骤3: 记录关键帧位姿信息 ==========
        KfCloudPose->push_back(myTf(q, p).Pose6D(t)); // 添加6D位姿（位置+姿态+时间戳）
        KfCloudPose->points.back().intensity = KfCloudPose->size()-1; // 使用强度字段存储关键帧ID

        // 调试信息输出（已注释）
        // for(int i = 0; i < KfCloudinB.size(); i++)
        //     printf("KF %d. Size: %d. %d\n",
        //             i, KfCloudinB[i]->size(), KfCloudinW[i]->size());

        // printf("Be4 add: GMap: %d.\n", globalMap->size(), KfCloudinW.back()->size());
        
        // ========== 步骤4: 更新全局地图 ==========
        // 将关键帧点云添加到全局地图中
        {
            lock_guard<mutex> lock(global_map_mtx); // 使用互斥锁保护全局地图，确保线程安全
            *globalMap += *KfCloudinW.back();       // 累加当前关键帧的世界坐标系点云到全局地图
        }

        // ========== 步骤5: 发送点云到地图处理队列 ==========
        // 将边际化点云发送到地图队列，用于surfel地图或其他地图结构的更新
        SendCloudToMapQueue(marginalizedCloudInW);

        // ========== 步骤6: 全局地图下采样过滤 ==========
        // 当关键帧数量大于1且启用地图发布时，对全局地图进行下采样以控制数据量
        if (KfCloudPose->size() > 1 && publish_map)
        {
            pcl::UniformSampling<PointXYZI> downsampler;  // 创建均匀下采样器
            downsampler.setRadiusSearch(leaf_size);       // 设置下采样半径
            downsampler.setInputCloud(globalMap);         // 设置输入为全局地图
            downsampler.filter(*globalMap);               // 执行下采样过滤，减少点云密度
        }

        // ========== 步骤7: 发布ROS话题 ==========
        // 7.1 发布边际化点云话题
        static ros::Publisher margcloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/marginalized_cloud", 100);
        Util::publishCloud(margcloud_pub, *marginalizedCloudInW, ros::Time(t), current_ref_frame);

        // 7.2 发布关键帧点云话题，并获取ROS消息用于后续封装
        sensor_msgs::PointCloud2 kfCloudROS
            = Util::publishCloud(kfcloud_pub, *KfCloudinW.back(), ros::Time(t), current_ref_frame);

        // 7.3 发布关键帧位姿点云话题，用于轨迹可视化
        sensor_msgs::PointCloud2 kfPoseCloudROS
            = Util::publishCloud(kfpose_pub, *KfCloudPose, ros::Time(t), current_ref_frame);
        
        // 7.4 封装并发布标准特征点云消息
        slict::FeatureCloud msg;  // 创建自定义的特征点云消息
        msg.header.stamp = ros::Time(t);    // 设置时间戳
        
        // 设置位姿信息（位置）
        msg.pose.position.x = p.x();
        msg.pose.position.y = p.y();
        msg.pose.position.z = p.z();
        
        // 设置位姿信息（姿态四元数）
        msg.pose.orientation.x = q.x();
        msg.pose.orientation.y = q.y();
        msg.pose.orientation.z = q.z();
        msg.pose.orientation.w = q.w();
        
        // 设置点云数据和时间信息        
        msg.extracted_cloud = kfCloudROS;    // 关键帧点云
        msg.edge_cloud = kfPoseCloudROS;     // 位姿轨迹点云
        msg.scanStartTime = t;               // 扫描开始时间
        msg.scanEndTime = t;                 // 扫描结束时间
        kfcloud_std_pub.publish(msg);        // 发布封装后的消息

        // 7.5 根据配置发布全局地图
        if (publish_map)
            Util::publishCloud(global_map_pub, *globalMap, ros::Time(t), current_ref_frame);

        tt_ufoupdate.Toc(); // 结束UFO地图更新计时    
    } // AdmitKeyframe() 函数结束

    void SendCloudToMapQueue(CloudXYZIPtr &cloud)
    {
        lock_guard<mutex> lg(mapqueue_mtx);
        mapqueue.push_back(cloud);
    }

    void UpdateMap()
    {
        while(ros::ok())
        {
            if (mapqueue.size() == 0)
            {
                this_thread::sleep_for(chrono::milliseconds(5));
                continue;
            }
            
            // Extract the cloud
            CloudXYZIPtr cloud;
            {
                lock_guard<mutex> lg(mapqueue_mtx);
                cloud = mapqueue.front();
                mapqueue.pop_front();
            }

            // Insert the cloud to the map
            {
                lock_guard<mutex> lg(map_mtx);
                if(use_ufm)
                    insertCloudToSurfelMap(*activeSurfelMap, *cloud);
                else
                {
                    if(activeikdtMap->Root_Node == nullptr)
                        activeikdtMap->Build(cloud->points);
                    else
                        activeikdtMap->Add_Points(cloud->points, true);
                }
            }
        }
    }

    /**
     * 点云与地图关联函数
     * 功能描述：将去畸变点云与地图进行关联，计算几何约束系数用于优化
     * @param Map UFO surfel地图引用，用于基于surfel的关联
     * @param activeikdtMap ikd-tree地图指针，用于基于k近邻的关联
     * @param tf_W_B 从传感器坐标系到世界坐标系的变换
     * @param CloudSkewed 原始带时间戳的畸变点云
     * @param CloudDeskewedDS 去畸变并下采样的点云
     * @param CloudCoef 输出的激光雷达约束系数向量
     * @param stat 关联统计信息映射（尺度->数量）
     * 主要步骤：
     *   1. 初始化系数缓冲区和关联器
     *   2. 并行处理每个点的坐标变换和有效性检查
     *   3. 根据地图类型执行不同的关联策略
     *   4. 统计关联结果
     */
    void AssociateCloudWithMap(ufoSurfelMap &Map, ikdtreePtr &activeikdtMap, mytf tf_W_B,
                               CloudXYZITPtr const &CloudSkewed, CloudXYZIPtr &CloudDeskewedDS,
                               vector<LidarCoef> &CloudCoef, map<int, int> &stat)
    {
        // ========== 步骤1: 初始化系数缓冲区 ==========
        int pointsCount = CloudDeskewedDS->points.size();
        if (CloudCoef.size() != pointsCount)
        {
            // 初始化约束系数缓冲区，预分配内存以提高性能
            CloudCoef.reserve(pointsCount);
        }

        // 创建静态点到地图关联器（避免重复创建开销）
        static PointToMapAssoc pma(nh_ptr);

        // ========== 步骤2: 并行处理每个特征点 ==========
        int featureTotal = CloudDeskewedDS->size();
        #pragma omp parallel for num_threads(MAX_THREADS)  // 使用OpenMP并行加速
        for(int k = 0; k < featureTotal; k++)
        {
            auto &point = CloudDeskewedDS->points[k];  // 当前处理的去畸变点
            int  point_idx = (int)(point.intensity);   // 原始点云中的索引（存储在强度字段中）
            int  coeff_idx = k;                        // 系数数组中的索引

            // ========== 2.1: 重置并初始化约束系数 ==========
            CloudCoef[coeff_idx].t = -1;                                    // 时间戳（-1表示未关联）
            CloudCoef[coeff_idx].t_ = CloudSkewed->points[point_idx].t;     // 原始时间戳
            CloudCoef[coeff_idx].d2P = -1;                                  // 点到平面距离（-1表示未计算）
            CloudCoef[coeff_idx].ptIdx = point_idx;                         // 原始点索引
            CloudCoef[coeff_idx].marginalized = false;                      // 边际化标志

            // ========== 2.2: 设置点的坐标表示 ==========
            PointXYZIT pointRaw = CloudSkewed->points[point_idx];          // 原始带时间戳的点
            PointXYZI  pointInB = point;                                   // 传感器坐标系（Body）中的点
            PointXYZI  pointInW = Util::transform_point(tf_W_B, pointInB); // 世界坐标系（World）中的点

            // ========== 2.3: 点有效性检查 ==========
            if(!Util::PointIsValid(pointInB) || pointRaw.t < 0)
            {
                // 如果点无效（NaN、Inf或时间戳无效），跳过处理
                // printf(KRED "Invalid surf point!: %f, %f, %f\n" RESET, pointInB.x, pointInB.y, pointInB.z);
                pointInB.x = 0; pointInB.y = 0; pointInB.z = 0; pointInB.intensity = 0;
                continue;
            }

            // ========== 2.4: 边际化检查（已注释的代码段） ==========
            // 以下代码用于检查包含节点是否可以边际化点，目前已被注释掉
            // // 创建包含谓词：检查深度和包含关系
            // auto containPred = ufopred::DepthE(surfel_min_depth)
            //                 && ufopred::Contains(ufoPoint3(pointInW.x, pointInW.y, pointInW.z));

            // // 查找包含该点的节点列表
            // deque<ufoNode> containingNode;
            // for (const ufoNode &node : Map.queryBV(containPred))
            //     containingNode.push_back(node);

            // // 如果点没有包含节点，将其视为可边际化点
            // if (containingNode.size() == 0)
            // {
            //     CloudCoef[coeff_idx + surfel_min_depth].marginalized = true;
            //     CloudCoef[coeff_idx + surfel_min_depth].finW = Vector3d(pointInW.x, pointInW.y, pointInW.z);
            // }
            // else
            // {
            //     // 遍历包含节点，检查surfel点数是否满足边际化条件
            //     for (const ufoNode &node : containingNode)
            //     {
            //         if (Map.getSurfel(containingNode.front()).getNumPoints() < surfel_min_point)
            //         {
            //             ROS_ASSERT( node.depth() == surfel_min_depth );
            //             CloudCoef[coeff_idx + surfel_min_depth].marginalized = true;
            //             CloudCoef[coeff_idx + surfel_min_depth].finW = Vector3d(pointInW.x, pointInW.y, pointInW.z);
            //         }
            //     }
            // }

            // ========== 2.5: 根据地图类型执行不同的关联策略 ==========
            if(use_ufm)
            {
                // 策略A: 使用UFO surfel地图进行关联
                Vector3d finB(pointInB.x, pointInB.y, pointInB.z);  // 传感器坐标系中的点
                Vector3d finW(pointInW.x, pointInW.y, pointInW.z);  // 世界坐标系中的点
                // 调用专用关联器进行基于surfel的点-地图关联
                pma.AssociatePointWithMap(pointRaw, finB, finW, Map, CloudCoef[coeff_idx]);
            }
            else
            {
                // 策略B: 使用ikd-tree地图进行基于k近邻的关联
                int numNbr = surfel_min_point;     // 需要查找的近邻点数量
                ikdtPointVec nbrPoints;            // 存储近邻点
                vector<float> knnSqDis;            // 存储到近邻点的平方距离
                
                // 在ikd-tree中进行k近邻搜索
                activeikdtMap->Nearest_Search(pointInW, numNbr, nbrPoints, knnSqDis);

                // ========== 2.5.1: 近邻点数量和距离检查 ==========
                if (nbrPoints.size() < numNbr)
                    continue;  // 近邻点不足，跳过该点
                else if (knnSqDis[numNbr - 1] > 5.0)
                    continue;  // 最远近邻点距离过大，跳过该点
                else
                {
                    // ========== 2.5.2: 平面拟合和约束生成 ==========
                    Vector4d pabcd;  // 平面方程系数 [a,b,c,d]: ax+by+cz+d=0
                    double rho;      // 平面拟合的可靠性评分
                    
                    // 对近邻点进行平面拟合
                    if(Util::fitPlane(nbrPoints, surfel_min_plnrty, dis_to_surfel_max, pabcd, rho))
                    {
                        // 计算点到平面的距离
                        float d2p = pabcd(0) * pointInW.x + pabcd(1) * pointInW.y + pabcd(2) * pointInW.z + pabcd(3);
                        
                        // 计算综合评分：考虑距离和可靠性
                        float score = (1 - 0.9 * fabs(d2p) / Util::pointDistance(pointInB))*rho;
                        // float score = 1 - 0.9 * fabs(d2p) / (1 + pow(Util::pointDistance(pointInB), 4)); // 备选评分方法
                        // float score = 1; // 简单评分方法

                        // ========== 2.5.3: 根据评分决定是否添加约束 ==========
                        if (score > score_min)
                        {
                            // 评分满足要求，添加到约束系数中
                            LidarCoef &coef = CloudCoef[coeff_idx];

                            coef.t      = pointRaw.t;                                        // 时间戳
                            coef.ptIdx  = point_idx;                                         // 点索引
                            coef.n      = pabcd;                                             // 平面法向量和距离
                            coef.scale  = surfel_min_depth;                                  // 尺度层级
                            coef.surfNp = numNbr;                                            // 近邻点数量
                            coef.plnrty = score;                                             // 平面度评分
                            coef.d2P    = d2p;                                               // 点到平面距离
                            coef.f      = Vector3d(pointRaw.x, pointRaw.y, pointRaw.z);     // 原始点坐标
                            coef.fdsk   = Vector3d(pointInB.x, pointInB.y, pointInB.z);     // 去畸变点坐标
                            coef.finW   = Vector3d(pointInW.x, pointInW.y, pointInW.z);     // 世界坐标系点坐标
                        }
                    }
                }
            }
        } // 并行循环结束

        // ========== 步骤3: 统计关联结果 ==========
        // 遍历所有处理过的特征点，统计成功关联的数量
        for(int i = 0; i < featureTotal; i++)
        {
            int point_idx = (int)CloudDeskewedDS->points[i].intensity;  // 获取原始点索引
            int coeff_idx = i;                                          // 系数索引

            auto &coef = CloudCoef[coeff_idx];  // 获取对应的约束系数
            if (coef.t >= 0)  // 如果时间戳有效（>=0），表示该点成功与地图关联
            {
                // CloudCoef.push_back(coef);  // 备用代码：将系数添加到列表（已注释）
                stat[coef.scale] += 1;       // 在统计映射中增加对应尺度的计数
                // break;                    // 备用代码：跳出循环（已注释）
            }
        }
    } // AssociateCloudWithMap() 函数结束
    
    void makeDVAReport(deque<map<int, int>> &stats, map<int, int> &DVA, int &total, string &DVAReport)
    {
        DVA.clear();
        total = 0;
        DVAReport = "";

        for(auto &stepdva : stats)
        {
            for(auto &dva : stepdva)
            {
                total += dva.second;
                DVA[dva.first] += dva.second;
            }
        }
        
        // Calculate the mean and variance of associations at each scale
        // double N = DVA.size();
        // double mean = total / N;
        // double variance = 0;
        // for(auto &dva : DVA)
        //     variance += std::pow(dva.second - mean, 2);

        // variance = sqrt(variance/N);

        // // Find the depths with association count within 2 variance
        // map<int, bool> inlier;
        // for(auto &dva : DVA)
        //     inlier[dva.first] = fabs(dva.second - mean) < variance;

        int max_depth = -1;
        int max_assoc = 0;
        for(auto &dva : DVA)
        {
            if (dva.second > max_assoc)
            {
                max_depth = dva.first;
                max_assoc = dva.second;
            }
        }            

        // Create a report with color code
        for(auto &dva : DVA)
            DVAReport += myprintf("%s[%2d, %5d]"RESET, dva.second > max_assoc/3 ? KYEL : KWHT, dva.first, dva.second);

        DVAReport += myprintf(". DM: %2d. MaxA: %d", max_depth, max_assoc);
    }
    
    /**
     * 因子选择函数
     * 功能描述：从滑动窗口中选择用于优化的IMU因子和激光雷达因子，控制优化问题规模
     * @param traj B样条轨迹对象，用于时间有效性检查和索引计算
     * @param imuSelected 输出的选中IMU因子索引向量
     * @param featureSelected 输出的选中激光雷达特征因子索引向量
     * 主要步骤：
     *   1. 选择所有有效的IMU因子
     *   2. 选择激光雷达因子（应用下采样）
     *   3. 如果激光雷达因子过多，进行随机下采样
     * 设计目的：平衡优化精度与计算效率，避免因子数量过多导致的性能问题
     */
    void FactorSelection(PoseSplineX &traj, vector<ImuIdx> &imuSelected, vector<lidarFeaIdx> &featureSelected)
    {
        // 每个节点上的耦合计数器（已注释，用于调试和分析）
        // vector<int> knot_count_imu(traj.numKnots(), 0);  // IMU因子在各节点的分布计数
        // vector<int> knot_count_ldr(traj.numKnots(), 0);  // 激光雷达因子在各节点的分布计数

        // ========== 步骤1: 选择IMU因子 ==========
        // 遍历滑动窗口中的所有IMU数据，选择有效的IMU因子
        for(int i = 0; i < WINDOW_SIZE; i++)        // 遍历滑动窗口中的每一帧
        {
            for(int j = 0; j < N_SUB_SEG; j++)      // 遍历每帧的所有时间子段
            {
                for(int k = 1; k < SwImuBundle[i][j].size(); k++)  // 遍历每个子段中的IMU样本（跳过第0个）
                {
                    double sample_time = SwImuBundle[i][j][k].t;  // 获取IMU样本的时间戳
                    
                    // 检查样本时间是否超出轨迹有效范围
                    if (!traj.TimeIsValid(sample_time, 1e-6))
                        continue;  // 时间无效，跳过该样本

                    auto us = traj.computeTIndex(sample_time);  // 计算时间索引
                    int knot_idx = us.second;                   // 获取对应的节点索引

                    // knot_count_imu[knot_idx] += 1;  // 节点计数（已注释）
                    imuSelected.push_back(ImuIdx(i, j, k));     // 添加到选中的IMU因子列表
                }
            }
        }

        // ========== 步骤2: 选择激光雷达因子 ==========
        // 按滑动窗口步骤分组的特征临时容器，用于进一步下采样
        vector<vector<lidarFeaIdx>> featureBySwStep(WINDOW_SIZE);

        // 选择激光雷达因子
        int total_selected = 0;  // 总选中因子计数
        for (int i = 0; i < WINDOW_SIZE; i++)  // 遍历滑动窗口中的每一帧
        {
            for (int k = 0; k < SwCloudDskDS[i]->size(); k++)  // 遍历当前帧中的每个点
            {
                // 应用下采样策略：计算了很多因子，但由于时间限制只使用其中一部分进行优化
                // 通过添加计数器来打乱因子顺序，让所有因子都有被使用的机会
                if ((k + i) % lidar_ds_rate != 0)
                    continue;  // 跳过不满足下采样条件的点

                auto &point = SwCloudDskDS[i]->points[k];        // 当前处理的点
                int  point_idx = (int)(point.intensity);         // 原始点云中的索引
                int  coeff_idx = k;                              // 系数数组中的索引

                LidarCoef &coef = SwLidarCoef[i][coeff_idx];     // 获取对应的激光雷达约束系数

                // 检查约束系数是否有效（t<0表示该点未成功关联到地图）
                if (coef.t < 0)
                    continue;

                double sample_time = coef.t;  // 获取样本时间戳

                // 检查样本时间是否在轨迹有效范围内
                if (!traj.TimeIsValid(sample_time, 1e-6))
                    continue;

                auto us = traj.computeTIndex(sample_time);  // 计算时间索引
                int knot_idx = us.second;                   // 获取对应的节点索引

                total_selected++;  // 增加总选中计数
                // knot_count_ldr[knot_idx] += 1;  // 节点计数（已注释）
                
                // 将特征索引添加到对应滑动窗口步骤的容器中
                featureBySwStep[i].push_back(lidarFeaIdx(i, k, coef.scale, total_selected));
            }
        }

        // ========== 步骤3: 激光雷达因子数量控制 ==========
        // 如果激光雷达特征数量仍然很大，随机选择一个子集
        if (total_selected > max_lidar_factor)
        {
            // ========== 3.1: 定义Fisher-Yates随机打乱算法 ==========
            // 使用Lambda函数实现经典的Fisher-Yates洗牌算法，确保均匀随机分布
            auto fisherYatesShuffle = [](std::vector<int>& array)
            {
                std::random_device rd;    // 硬件随机数生成器
                std::mt19937 gen(rd());   // 梅森旋转伪随机数生成器

                // 从数组末尾开始，逐个与前面的随机位置交换
                for (int i = array.size() - 1; i > 0; --i)
                {
                    std::uniform_int_distribution<int> distribution(0, i);  // 均匀分布[0,i]
                    int j = distribution(gen);  // 生成随机索引
                    std::swap(array[i], array[j]);  // 交换元素
                }
            };

            // ========== 3.2: 计算每个滑动窗口步骤需要的特征数量 ==========
            int maxFeaPerSwStep = ceil(double(max_lidar_factor) / WINDOW_SIZE);

            // ========== 3.3: 创建打乱索引容器 ==========
            // vector<vector<lidarFeaIdx>> featureBySwStepShuffled(WINDOW_SIZE);  // 备用容器（已注释）
            vector<vector<int>> shuffledIdx(WINDOW_SIZE);  // 存储打乱后的索引

            // ========== 3.4: 并行生成打乱索引 ==========
            #pragma omp parallel for num_threads(MAX_THREADS)  // 使用OpenMP并行处理
            for(int wid = 0; wid < WINDOW_SIZE; wid++)
            {
                // 为当前窗口步骤创建索引数组
                shuffledIdx[wid] = vector<int>(featureBySwStep[wid].size());
                std::iota(shuffledIdx[wid].begin(), shuffledIdx[wid].end(), 0);  // 初始化为0,1,2,3,...
                
                // 多次打乱以增强随机性（提高随机分布质量）
                fisherYatesShuffle(shuffledIdx[wid]);
                fisherYatesShuffle(shuffledIdx[wid]);
                fisherYatesShuffle(shuffledIdx[wid]);
            }

            // ========== 3.5: 根据打乱后的索引选择特征 ==========
            for(int wid = 0; wid < WINDOW_SIZE; wid++)
                for(int idx = 0; idx < min(maxFeaPerSwStep, (int)featureBySwStep[wid].size()); idx++)
                        featureSelected.push_back(featureBySwStep[wid][shuffledIdx[wid][idx]]);
        }
        else
        {
            // ========== 步骤4: 直接添加所有特征 ==========
            // 如果特征数量在限制范围内，直接添加所有选中的特征
            for(int wid = 0; wid < WINDOW_SIZE; wid++)
                for(int idx = 0; idx < featureBySwStep[wid].size(); idx++)
                    featureSelected.push_back(featureBySwStep[wid][idx]);
        }
    } // FactorSelection() 函数结束

    void DetectLoop()
    {
        // For visualization
        static ros::Publisher loop_kf_nbr_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_kf_nbr", 100);
        CloudPosePtr loopKfNbr(new CloudPose());

        static ros::Publisher loop_currkf_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_currkf", 100);
        CloudPosePtr loopCurrKf(new CloudPose());

        static ros::Publisher loop_prevkf_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_prevkf", 100);
        CloudPosePtr loopPrevKf(new CloudPose());

        static ros::Publisher loop_currCloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_curr_cloud", 100);
        static ros::Publisher loop_prevCloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_prev_cloud", 100);
        static ros::Publisher loop_currCloud_refined_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_curr_refined_cloud", 100);

        // Extract the current pose
        int currPoseId = (int)(KfCloudPose->points.back().intensity);
        PointPose currPose = KfCloudPose->points[currPoseId];
        CloudXYZIPtr currCloudInB(new CloudXYZI()); CloudXYZIPtr currCloudInW(new CloudXYZI());
        *currCloudInB = *KfCloudinB[currPoseId];
        *currCloudInW = *KfCloudinW[currPoseId];

        // Search for the nearest neighbours
        vector<int> knn_idx(loop_kf_nbr); vector<float> knn_sq_dis(loop_kf_nbr);
        static KdTreeFLANN<PointPose> kdTreeKeyFrames;
        kdTreeKeyFrames.setInputCloud(KfCloudPose);
        kdTreeKeyFrames.nearestKSearch(currPose, loop_kf_nbr, knn_idx, knn_sq_dis);

        // Publish the current keyframe
        loopCurrKf->push_back(currPose);
        if (loopCurrKf->size() > 0)
            Util::publishCloud(loop_currkf_pub, *loopCurrKf, ros::Time(currPose.t), current_ref_frame);

        // Find the oldest index in the neigborhood
        int prevPoseId = -1;
        PointPose prevPose;
        CloudXYZIPtr prevCloudInW(new CloudXYZI());

        for (auto nbr_idx : knn_idx)
        {
            PointPose &kfPose = KfCloudPose->points[nbr_idx];
            loopKfNbr->push_back(kfPose);

            ROS_ASSERT(nbr_idx == (int)(kfPose.intensity));
            if (prevPoseId == -1 || nbr_idx < prevPoseId)
                prevPoseId = nbr_idx;
        }

        // Publish the nbr kf for visualization
        if (loopKfNbr->size() > 0)
            Util::publishCloud(loop_kf_nbr_pub, *loopKfNbr, ros::Time(currPose.t), current_ref_frame);

        static int LAST_KF_COUNT = KfCloudPose->size();

        // Only do the check every 5 keyframes
        int newKfCount = KfCloudPose->size();
        if (newKfCount - LAST_KF_COUNT < 5 || newKfCount <= loop_kf_nbr)
            return;
        LAST_KF_COUNT = newKfCount;

        // If new loop is too close to last loop in time, skip
        if (!loopPairs.empty())
        {
            double time_since_lastloop = fabs(KfCloudPose->points.back().t - KfCloudPose->points[loopPairs.back().currPoseId].t);
            // printf("Time since last loop: %f\n", time_since_lastloop);

            if (time_since_lastloop < loop_time_mindiff)
                return;
        }

        double time_nbr_diff = fabs(KfCloudPose->points[currPoseId].t - KfCloudPose->points[prevPoseId].t);
        // printf("Time nbr diff: %f\n", time_nbr_diff);

        // Return if no neighbour found, or the two poses are too close in time
        if (prevPoseId == -1 || time_nbr_diff < loop_time_mindiff || abs(currPoseId - prevPoseId) < loop_kf_nbr)
            return;
        else
            prevPose = KfCloudPose->points[prevPoseId];

        // Previous pose detected, build the previous local map

        // Find the range of keyframe Ids
        int bId = prevPoseId; int fId = prevPoseId; int span = fId - bId;
        while(span < loop_kf_nbr)
        {
            bId = max(0, bId - 1);
            fId = min(fId + 1, currPoseId - 1);

            int new_span = fId - bId;

            if ( new_span == span || new_span >= loop_kf_nbr )
                break;
            else
                span = new_span;
        }

        // Extract the keyframe pointcloud around the reference pose
        for(int kfId = bId; kfId < fId; kfId++)
        {
            loopPrevKf->push_back(KfCloudPose->points[kfId]);
            *prevCloudInW += *KfCloudinW[kfId];
        }

        // Publish previous keyframe for vizualization
        if (loopPrevKf->size() > 0)
            Util::publishCloud(loop_prevkf_pub, *loopPrevKf, ros::Time(currPose.t), current_ref_frame);

        // Downsample the pointclouds
        pcl::UniformSampling<PointXYZI> downsampler;
        double voxel_size = max(leaf_size, 0.4);
        downsampler.setRadiusSearch(voxel_size);

        downsampler.setInputCloud(prevCloudInW);
        downsampler.filter(*prevCloudInW);
        
        downsampler.setInputCloud(currCloudInB);
        downsampler.filter(*currCloudInB);

        // Publish the cloud for visualization
        Util::publishCloud(loop_prevCloud_pub, *prevCloudInW, ros::Time(currPose.t), current_ref_frame);
        Util::publishCloud(loop_currCloud_pub, *currCloudInW, ros::Time(currPose.t), current_ref_frame);

        // Check match by ICP
        myTf tf_W_Bcurr_start = myTf(currPose);
        myTf tf_W_Bcurr_final = tf_W_Bcurr_start; Matrix4f tfm_W_Bcurr_final;

        bool icp_passed = false; double icpFitnessRes = -1; double icpCheckTime = -1;
        icp_passed =    CheckICP(prevCloudInW, currCloudInB,
                                 tf_W_Bcurr_start.cast<float>().tfMat(), tfm_W_Bcurr_final,
                                 histDis, icpMaxIter, icpFitnessThres, icpFitnessRes, icpCheckTime);
        lastICPFn = icpFitnessRes;

        // Return if icp check fails
        if (!icp_passed)
            return;

        tf_W_Bcurr_final = myTf(tfm_W_Bcurr_final).cast<double>();

        printf("%sICP %s. T_W(%03d)_B(%03d). Fn: %.3f. icpTime: %.3f.\n"
               "Start: Pos: %f, %f, %f. YPR: %f, %f, %f\n"
               "Final: Pos: %f, %f, %f. YPR: %f, %f, %f\n"
                RESET,
                icp_passed ? KBLU : KRED, icp_passed ? "passed" : "failed", prevPoseId, currPoseId, icpFitnessRes, icpCheckTime,
                tf_W_Bcurr_start.pos.x(), tf_W_Bcurr_start.pos.y(), tf_W_Bcurr_start.pos.z(),
                tf_W_Bcurr_start.yaw(),   tf_W_Bcurr_start.pitch(), tf_W_Bcurr_start.roll(),
                tf_W_Bcurr_final.pos.x(), tf_W_Bcurr_final.pos.y(), tf_W_Bcurr_final.pos.z(),
                tf_W_Bcurr_final.yaw(),   tf_W_Bcurr_final.pitch(), tf_W_Bcurr_final.roll());

        // Add the loop to buffer
        loopPairs.push_back(LoopPrior(prevPoseId, currPoseId, 1e-3, icpFitnessRes,
                                      mytf(prevPose).inverse()*tf_W_Bcurr_final));

        // Publish the transform current cloud
        pcl::transformPointCloud(*currCloudInB, *currCloudInW, tf_W_Bcurr_final.pos, tf_W_Bcurr_final.rot);
        Util::publishCloud(loop_currCloud_refined_pub, *currCloudInW, ros::Time(currPose.t), current_ref_frame);
    }

    bool CheckICP(CloudXYZIPtr &ref_pcl, CloudXYZIPtr &src_pcl, Matrix4f relPosIcpGuess, Matrix4f &relPosIcpEst,
                  double hisKFSearchRadius, int icp_max_iters, double icpFitnessThres, double &icpFitnessRes, double &ICPtime)
    {

        /* #region Calculate the relative pose constraint ---------------------------------------------------------------*/

        TicToc tt_icp;

        pcl::IterativeClosestPoint<PointXYZI, PointXYZI> icp;
        icp.setMaxCorrespondenceDistance(hisKFSearchRadius * 2);
        icp.setMaximumIterations(icp_max_iters);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);
        
        icp.setInputSource(src_pcl);
        icp.setInputTarget(ref_pcl);

        CloudXYZIPtr aligned_result(new CloudXYZI());

        icp.align(*aligned_result, relPosIcpGuess);

        bool icp_passed   = false;
        bool icpconverged = icp.hasConverged();
        icpFitnessRes     = icp.getFitnessScore();
        relPosIcpEst      = icp.getFinalTransformation();

        ICPtime = tt_icp.Toc();

        if (!icpconverged || icpFitnessRes > icpFitnessThres)
        {
            // if (extended_report)
            // {
            //     printf(KRED "\tICP time: %9.3f ms. ICP %s. Fitness: %9.3f, threshold: %3.1f\n" RESET,
            //     tt_icp.GetLastStop(),
            //     icpconverged ? "converged" : "fails to converge",
            //     icpFitnessRes, icpFitnessThres);
            // }
        }
        else
        {
            // if (extended_report)
            // {
            //     printf(KBLU "\tICP time: %9.3f ms. ICP %s. Fitness: %9.3f, threshold: %3.1f\n" RESET,
            //         tt_icp.GetLastStop(),
            //         icpconverged ? "converged" : "fails to converge",
            //         icpFitnessRes, icpFitnessThres);
            // }

            icp_passed = true;
        }

        return icp_passed;

        /* #endregion Calculate the relative pose constraint ------------------------------------------------------------*/        

    }

    void BundleAdjustment(BAReport &report)
    {
        static int LAST_LOOP_COUNT = loopPairs.size();
        int newLoopCount = loopPairs.size();

        // Return if no new loop detected
        if (newLoopCount - LAST_LOOP_COUNT < 1)
            return;
        LAST_LOOP_COUNT = newLoopCount;

        // Solve the pose graph optimization problem
        OptimizePoseGraph(KfCloudPose, loopPairs, report);

        TicToc tt_rebuildmap;

        // Recompute the keyframe pointclouds
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int i = 0; i < KfCloudPose->size(); i++)
        {
            myTf tf_W_B(KfCloudPose->points[i]);
            pcl::transformPointCloud(*KfCloudinB[i], *KfCloudinW[i], tf_W_B.pos, tf_W_B.rot);
        }

        // Recompute the globalmap and ufomap
        {
            lock_guard<mutex> lggm(global_map_mtx);
            globalMap->clear();

            for(int i = 0; i < KfCloudPose->size(); i++)
                *globalMap += *KfCloudinW[i];

            // Downsample the global map
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setRadiusSearch(leaf_size);
            downsampler.setInputCloud(globalMap);
            downsampler.filter(*globalMap);

            Util::publishCloud(global_map_pub, *globalMap, ros::Time(KfCloudPose->points.back().t), current_ref_frame);

            // Clear the map queu            
            {
                lock_guard<mutex> lgmq(mapqueue_mtx);
                mapqueue.clear();                       // TODO: Should transform the remaining clouds to the new coordinates.
            }
            // Build the surfelmap
            {
                lock_guard<mutex> lgam(map_mtx);

                if(use_ufm)
                {
                    activeSurfelMap->clear();
                    insertCloudToSurfelMap(*activeSurfelMap, *globalMap);
                }
                else
                {   
                    activeikdtMap = ikdtreePtr(new ikdtree(0.5, 0.6, leaf_size));
                    activeikdtMap->Add_Points(globalMap->points, false);
                }
            }

            // Increment the ufomap version
            ufomap_version++;
        }

        tt_rebuildmap.Toc();

        report.rebuildmap_time = tt_rebuildmap.GetLastStop();
    }

    void OptimizePoseGraph(CloudPosePtr &kfCloud, const deque<LoopPrior> &loops, BAReport &report)
    {
        TicToc tt_pgopt;

        static int BA_NUM = -1;

        int KF_NUM = kfCloud->size();

        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = omp_get_max_threads();
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Create params and load data
        double **PARAM_POSE = new double *[KF_NUM];
        for(int i = 0; i < KF_NUM; i++)
        {
            PARAM_POSE[i] = new double[7];

            PARAM_POSE[i][0] = kfCloud->points[i].x;
            PARAM_POSE[i][1] = kfCloud->points[i].y;
            PARAM_POSE[i][2] = kfCloud->points[i].z;
            PARAM_POSE[i][3] = kfCloud->points[i].qx;
            PARAM_POSE[i][4] = kfCloud->points[i].qy;
            PARAM_POSE[i][5] = kfCloud->points[i].qz;
            PARAM_POSE[i][6] = kfCloud->points[i].qw;

            ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
            problem.AddParameterBlock(PARAM_POSE[i], 7, local_parameterization);

            // Fix the last pose
            if (i == KF_NUM - 1)
                problem.SetParameterBlockConstant(PARAM_POSE[i]);
        }

        // Add relative pose factors
        vector<ceres::internal::ResidualBlock *> res_ids_relpose;
        double cost_relpose_init = -1, cost_relpose_final = -1;
        for(int i = 1; i < KF_NUM; i++)
        {
            for (int j = 1; j < rib_edge; j++)
            {
                int jj = j;

                // Make an edge to the first pose for the poses with 5 steps
                if (i - j <= 0)
                    jj = i;

                myTf pose_i = myTf(kfCloud->points[i]);
                myTf pose_j = myTf(kfCloud->points[i-jj]);

                RelOdomFactor* relodomfactor = new RelOdomFactor(pose_i.pos, pose_j.pos, pose_i.rot, pose_j.rot,
                                                                 odom_q_noise, odom_p_noise);
                ceres::internal::ResidualBlock *res_id =  problem.AddResidualBlock(relodomfactor, NULL, PARAM_POSE[i], PARAM_POSE[i-jj]);
                res_ids_relpose.push_back(res_id);
            }
        }

        // Add loop factors
        vector<ceres::internal::ResidualBlock *> res_ids_loop;
        double cost_loop_init = -1, cost_loop_final = -1;
        for(auto &loop_edge : loopPairs)
        {
            // printf("Loop Factor: prev %d, curr: %d\n", loop_edge.prev_idx, loop_edge.curr_idx);
            
            int &curr_idx = loop_edge.currPoseId;
            int &prev_idx = loop_edge.prevPoseId;

            double &JKavr = loop_edge.JKavr;
            double &IcpFn = loop_edge.IcpFn;

            myTf pose_i = myTf(kfCloud->points[prev_idx]);
            myTf pose_j = myTf(kfCloud->points[prev_idx])*loop_edge.tf_Bp_Bc;

            RelOdomFactor* relodomfactor = new RelOdomFactor(pose_i.pos, pose_j.pos, pose_i.rot, pose_j.rot,
                                                             odom_q_noise*loop_weight, odom_p_noise*loop_weight);
            ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(relodomfactor, NULL, PARAM_POSE[prev_idx], PARAM_POSE[curr_idx]);
            res_ids_loop.push_back(res_id);
        }
        
        Util::ComputeCeresCost(res_ids_relpose, cost_relpose_init, problem);
        Util::ComputeCeresCost(res_ids_loop, cost_loop_init, problem);
        
        ceres::Solve(options, &problem, &summary);

        Util::ComputeCeresCost(res_ids_relpose, cost_relpose_final, problem);
        Util::ComputeCeresCost(res_ids_loop, cost_loop_final, problem);

        // Return the keyframe result
        for(int i = 0; i < KF_NUM; i++)
        {
            kfCloud->points[i].x  = PARAM_POSE[i][0];
            kfCloud->points[i].y  = PARAM_POSE[i][1];
            kfCloud->points[i].z  = PARAM_POSE[i][2];
            kfCloud->points[i].qx = PARAM_POSE[i][3];
            kfCloud->points[i].qy = PARAM_POSE[i][4];
            kfCloud->points[i].qz = PARAM_POSE[i][5];
            kfCloud->points[i].qw = PARAM_POSE[i][6];
        }
        
        baReport.turn           = (BA_NUM++);
        baReport.pgopt_time     = tt_pgopt.Toc();
        baReport.pgopt_iter     = summary.iterations.size();
        baReport.factor_relpose = res_ids_relpose.size();
        baReport.factor_loop    = res_ids_loop.size();
        baReport.J0             = summary.initial_cost;
        baReport.JK             = summary.final_cost;
        baReport.J0_relpose     = cost_relpose_init;
        baReport.JK_relpose     = cost_relpose_final;
        baReport.J0_loop        = cost_loop_init;
        baReport.JK_loop        = cost_loop_final;
    }

    void VisualizeLoop()
    {
        // Visualize the loop
        static visualization_msgs::Marker loop_marker; static bool loop_marker_inited = false;
        static ros::Publisher loop_marker_pub = nh_ptr->advertise<visualization_msgs::Marker>("/loop_marker", 100);
        static std_msgs::ColorRGBA color;

        if (!loop_marker_inited)
        {
            // Set up the loop marker
            loop_marker_inited = true;
            loop_marker.header.frame_id = current_ref_frame;
            loop_marker.ns       = "loop_marker";
            loop_marker.type     = visualization_msgs::Marker::LINE_LIST;
            loop_marker.action   = visualization_msgs::Marker::ADD;
            loop_marker.pose.orientation.w = 1.0;
            loop_marker.lifetime = ros::Duration(0);
            loop_marker.id       = 0;

            loop_marker.scale.x = 0.3; loop_marker.scale.y = 0.3; loop_marker.scale.z = 0.3;
            loop_marker.color.r = 0.0; loop_marker.color.g = 1.0; loop_marker.color.b = 1.0; loop_marker.color.a = 1.0;
            
            color.r = 0.0; color.g = 1.0; color.b = 1.0; color.a = 1.0;
        }

        loop_marker.points.clear();
        loop_marker.colors.clear();
        for(int i = 0; i < loopPairs.size(); i++)
        {
            int curr_idx = loopPairs[i].currPoseId;
            int prev_idx = loopPairs[i].prevPoseId;

            auto pose_curr = KfCloudPose->points[curr_idx];
            auto pose_prev = KfCloudPose->points[prev_idx];

            // Updating the line segments------------------------
            
            geometry_msgs::Point point;

            point.x = pose_curr.x;
            point.y = pose_curr.y;
            point.z = pose_curr.z;

            loop_marker.points.push_back(point);
            loop_marker.colors.push_back(color);

            point.x = pose_prev.x;
            point.y = pose_prev.y;
            point.z = pose_prev.z;

            loop_marker.points.push_back(point);
            loop_marker.colors.push_back(color);
        }
        // Publish the loop markers
        loop_marker_pub.publish(loop_marker);
    }

    void VisualizeSwTraj()
    {
        // Publish the sliding window trajectory and log the spline in the world frame
        {
            // Publish the traj
            static ros::Publisher swprop_viz_pub = nh_ptr->advertise<nav_msgs::Path>("/swprop_traj", 100);

            // Check if we have completed relocalization
            myTf tf_L0_Lprior(Quaternd(1, 0, 0, 0), Vector3d(0, 0, 0));
            if(reloc_stat == RELOCALIZED)
                tf_L0_Lprior = tf_Lprior_L0.inverse();

            // static ofstream swtraj_log;
            // static bool one_shot = true;
            // if (one_shot)
            // {
            //     swtraj_log.precision(std::numeric_limits<double>::digits10 + 1);
            //     swtraj_log.open((log_dir + "/swtraj.csv").c_str());
            //     swtraj_log.close(); // To reset the file
            //     one_shot = false;
            // }

            // // Append the data
            // swtraj_log.open((log_dir + "/swtraj.csv").c_str(), std::ios::app);

            double time_stamp = SwTimeStep.back().back().final_time;

            // Publish the propagated poses
            nav_msgs::Path prop_path;
            prop_path.header.frame_id = slam_ref_frame;
            prop_path.header.stamp = ros::Time(time_stamp);
            for(int i = 0; i < WINDOW_SIZE; i++)
            {
                for(int j = 0; j < SwPropState[i].size(); j++)
                {
                    for (int k = 0; k < SwPropState[i][j].size(); k++)
                    {
                        geometry_msgs::PoseStamped msg;
                        msg.header.frame_id = slam_ref_frame;
                        msg.header.stamp = ros::Time(SwPropState[i][j].t[k]);
                        
                        Vector3d pInL0 = tf_L0_Lprior*SwPropState[i][j].P[k];
                        msg.pose.position.x = pInL0.x();
                        msg.pose.position.y = pInL0.y();
                        msg.pose.position.z = pInL0.z();
                        
                        prop_path.poses.push_back(msg);

                        if (i == 0)
                        {
                            SE3d pose = GlobalTraj->pose(SwPropState[i][j].t[k]);
                            Vector3d pos = pose.translation();
                            Quaternd qua = pose.so3().unit_quaternion();
                            Vector3d vel = GlobalTraj->transVelWorld(SwPropState[i][j].t[k]);
                            Vector3d gyr = GlobalTraj->rotVelBody(SwPropState[i][j].t[k]) + sfBig[i][j];
                            Vector3d acc = qua.inverse()*(GlobalTraj->transAccelWorld(SwPropState[i][j].t[k]) + GRAV) + sfBia[i][j];

                            // swtraj_log << SwPropState[i][j].t[k]
                            //            << "," << SwPropState[i][j].P[k].x() << "," << SwPropState[i][j].P[k].y() << "," << SwPropState[i][j].P[k].z()
                            //            << "," << SwPropState[i][j].Q[k].x() << "," << SwPropState[i][j].Q[k].y() << "," << SwPropState[i][j].Q[k].z() << "," << SwPropState[i][j].Q[k].w()
                            //            << "," << SwPropState[i][j].V[k].x() << "," << SwPropState[i][j].V[k].y() << "," << SwPropState[i][j].V[k].z()
                            //            << "," << SwPropState[i][j].gyr[k].x() << "," << SwPropState[i][j].gyr[k].y() << "," << SwPropState[i][j].gyr[k].z()
                            //            << "," << SwPropState[i][j].acc[k].x() << "," << SwPropState[i][j].acc[k].y() << "," << SwPropState[i][j].acc[k].z()
                            //            << "," << pos.x() << "," << pos.y() << "," << pos.z()
                            //            << "," << qua.x() << "," << qua.y() << "," << qua.z() << "," << qua.w()
                            //            << "," << vel.x() << "," << vel.y() << "," << vel.z()
                            //            << "," << gyr.x() << "," << gyr.y() << "," << gyr.z()
                            //            << "," << acc.x() << "," << acc.y() << "," << acc.z()
                            //            << endl;
                        }
                    }
                }
            }
            // swtraj_log.close();
            swprop_viz_pub.publish(prop_path);
        }

        // Publish the control points
        {
            static ros::Publisher sw_ctr_pose_viz_pub = nh_ptr->advertise<nav_msgs::Path>("/sw_ctr_pose", 100);
            
            double SwTstart = SwTimeStep.front().front().start_time;
            double SwTfinal = SwTimeStep.front().front().final_time;

            double SwDur = SwTfinal - SwTstart;

            nav_msgs::Path path;
            path.header.frame_id = slam_ref_frame;
            path.header.stamp = ros::Time(SwTfinal);

            for(int knot_idx = GlobalTraj->numKnots() - 1; knot_idx >= 0; knot_idx--)
            {
                double tknot = GlobalTraj->getKnotTime(knot_idx);
                if (tknot < SwTstart - 2*SwDur )
                    break;

                Vector3d pos = GlobalTraj->getKnotPos(knot_idx);
                geometry_msgs::PoseStamped msg;
                msg.header.frame_id = slam_ref_frame;
                msg.header.stamp = ros::Time(tknot);
                
                msg.pose.position.x = pos.x();
                msg.pose.position.y = pos.y();
                msg.pose.position.z = pos.z();
                
                path.poses.push_back(msg);
            }

            sw_ctr_pose_viz_pub.publish(path);
        }

        // Publishing odometry stuff
        static myTf tf_Lprior_L0_init;
        {
            static bool one_shot = true;
            if (one_shot)
            {
                // Get the init transform
                vector<double> T_W_B_ = {1, 0, 0, 0,
                                         0, 1, 0, 0,
                                         0, 0, 1, 0,
                                         0, 0, 0, 1};
                nh_ptr->getParam("/T_M_W_init", T_W_B_);
                Matrix4d T_B_V = Matrix<double, 4, 4, RowMajor>(&T_W_B_[0]);
                tf_Lprior_L0_init = myTf(T_B_V);
                
                one_shot = false;
            }
        }

        // Stuff in world frame
        static ros::Publisher opt_odom_pub           = nh_ptr->advertise<nav_msgs::Odometry>("/opt_odom", 100);
        static ros::Publisher opt_odom_high_freq_pub = nh_ptr->advertise<nav_msgs::Odometry>("/opt_odom_high_freq", 100);
        static ros::Publisher lastcloud_pub          = nh_ptr->advertise<sensor_msgs::PointCloud2>("/lastcloud", 100);
        // Stuff in map frame
        static ros::Publisher opt_odom_inM_pub       = nh_ptr->advertise<nav_msgs::Odometry>("/opt_odom_inM", 100);
        
        if (reloc_stat != RELOCALIZED)
        {
            // Publish the odom
            PublishOdom(opt_odom_pub, sfPos.back().back(), sfQua.back().back(),
                        sfVel.back().back(), SwPropState.back().back().gyr.back(), SwPropState.back().back().acc.back(),
                        sfBig.back().back(), sfBia.back().back(), ros::Time(SwTimeStep.back().back().final_time), slam_ref_frame);

            // Publish the odom at sub segment ends
            for(int i = 0; i < N_SUB_SEG; i++)
            {
                double time_stamp = SwTimeStep.front()[i].final_time;
                PublishOdom(opt_odom_high_freq_pub, sfPos.front()[i], sfQua.front()[i],
                            sfVel.front()[i], SwPropState.front()[i].gyr.back(), SwPropState.front()[i].acc.back(),
                            sfBig.front()[i], sfBia.front()[i], ros::Time(time_stamp), slam_ref_frame);
            }

            // Publish the latest cloud
            CloudXYZIPtr latestCloud(new CloudXYZI());
            pcl::transformPointCloud(*SwCloudDsk.back(), *latestCloud, sfPos.back().back(), sfQua.back().back());
            Util::publishCloud(lastcloud_pub, *latestCloud, ros::Time(SwTimeStep.back().back().final_time), slam_ref_frame);
            
            // Publish pose in map frame by the initial pose guess
            Vector3d posInM = tf_Lprior_L0_init*sfPos.back().back();
            Quaternd quaInM = tf_Lprior_L0_init.rot*sfQua.back().back();
            Vector3d velInM = tf_Lprior_L0_init.rot*sfVel.back().back();  
            PublishOdom(opt_odom_inM_pub, posInM, quaInM,
                        velInM, SwPropState.back().back().gyr.back(), SwPropState.back().back().acc.back(),
                        sfBig.back().back(), sfBia.back().back(), ros::Time(SwTimeStep.back().back().final_time), "map");

            // Publish the transform between map and world at low rate
            {
                // static double update_time = -1;
                // if (update_time == -1 || ros::Time::now().toSec() - update_time > 1.0)
                static bool oneshot = true;
                if(oneshot)
                {
                    oneshot = false;
                    // update_time = ros::Time::now().toSec();
                    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
                    geometry_msgs::TransformStamped rostf_M_W;
                    rostf_M_W.header.stamp            = ros::Time::now();
                    rostf_M_W.header.frame_id         = "map";
                    rostf_M_W.child_frame_id          = slam_ref_frame;
                    rostf_M_W.transform.translation.x = tf_Lprior_L0_init.pos.x();
                    rostf_M_W.transform.translation.y = tf_Lprior_L0_init.pos.y();
                    rostf_M_W.transform.translation.z = tf_Lprior_L0_init.pos.z();
                    rostf_M_W.transform.rotation.x    = tf_Lprior_L0_init.rot.x();
                    rostf_M_W.transform.rotation.y    = tf_Lprior_L0_init.rot.y();
                    rostf_M_W.transform.rotation.z    = tf_Lprior_L0_init.rot.z();
                    rostf_M_W.transform.rotation.w    = tf_Lprior_L0_init.rot.w();
                    static_broadcaster.sendTransform(rostf_M_W);
                }
            }
        }
        else
        {
            static myTf tf_L0_Lprior = tf_Lprior_L0.inverse();

            // Publish the odom in the original slam reference frame
            Vector3d posInW = tf_L0_Lprior*sfPos.back().back();
            Quaternd quaInW = tf_L0_Lprior.rot*sfQua.back().back();
            Vector3d velInW = tf_L0_Lprior.rot*sfVel.back().back();        
            PublishOdom(opt_odom_pub, posInW, quaInW,
                        velInW, SwPropState.back().back().gyr.back(), SwPropState.back().back().acc.back(),
                        sfBig.back().back(), sfBia.back().back(), ros::Time(SwTimeStep.back().back().final_time), slam_ref_frame);

            // Publish the odom at sub segment ends in the original slam reference frame
            for(int i = 0; i < N_SUB_SEG; i++)
            {
                double time_stamp = SwTimeStep.front()[i].final_time;
                Vector3d posInW = tf_L0_Lprior*sfPos.front()[i];
                Quaternd quaInW = tf_L0_Lprior.rot*sfQua.front()[i];
                Vector3d velInW = tf_L0_Lprior.rot*sfVel.front()[i];        
                PublishOdom(opt_odom_high_freq_pub, posInW, quaInW,
                            velInW, SwPropState.front()[i].gyr.back(), SwPropState.front()[i].acc.back(),
                            sfBig.front()[i], sfBia.front()[i], ros::Time(time_stamp), slam_ref_frame);
            }

            // Publish the latest cloud in the original slam reference frame
            CloudXYZIPtr latestCloud(new CloudXYZI());
            pcl::transformPointCloud(*SwCloudDsk.back(), *latestCloud, posInW, quaInW);
            Util::publishCloud(lastcloud_pub, *latestCloud, ros::Time(SwTimeStep.back().back().final_time), slam_ref_frame);

            // Publish pose in map frame by the true transform
            PublishOdom(opt_odom_inM_pub, sfPos.back().back(), sfQua.back().back(),
                        sfVel.back().back(), SwPropState.back().back().gyr.back(), SwPropState.back().back().acc.back(),
                        sfBig.back().back(), sfBia.back().back(), ros::Time(SwTimeStep.back().back().final_time), current_ref_frame);

            // Publish the transform between map and world at low rate
            {
                // static double update_time = -1;
                // if (update_time == -1 || ros::Time::now().toSec() - update_time > 1.0)
                static bool oneshot = true;
                if(oneshot)
                {
                    oneshot = false;
                    // update_time = ros::Time::now().toSec();
                    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
                    geometry_msgs::TransformStamped rostf_M_W;
                    rostf_M_W.header.stamp            = ros::Time::now();
                    rostf_M_W.header.frame_id         = "map";
                    rostf_M_W.child_frame_id          = slam_ref_frame;
                    rostf_M_W.transform.translation.x = tf_Lprior_L0.pos.x();
                    rostf_M_W.transform.translation.y = tf_Lprior_L0.pos.y();
                    rostf_M_W.transform.translation.z = tf_Lprior_L0.pos.z();
                    rostf_M_W.transform.rotation.x    = tf_Lprior_L0.rot.x();
                    rostf_M_W.transform.rotation.y    = tf_Lprior_L0.rot.y();
                    rostf_M_W.transform.rotation.z    = tf_Lprior_L0.rot.z();
                    rostf_M_W.transform.rotation.w    = tf_Lprior_L0.rot.w();
                    static_broadcaster.sendTransform(rostf_M_W);
                }
            }
        }

        // Publish the relocalization status
        {
            static ros::Publisher reloc_stat_pub = nh_ptr->advertise<std_msgs::String>("/reloc_stat", 100);
            std_msgs::String msg;
            if (reloc_stat == NOT_RELOCALIZED)
                msg.data = "NOT_RELOCALIZED";
            else if (reloc_stat == RELOCALIZED)
                msg.data = "RELOCALIZED";
            else if (reloc_stat == RELOCALIZING)
                msg.data = "RELOCALIZING";
            reloc_stat_pub.publish(msg);
        }

        {
            static ros::Publisher reloc_pose_pub = nh_ptr->advertise<std_msgs::String>("/reloc_pose_str", 100);
            std_msgs::String msg;
            if (reloc_stat == NOT_RELOCALIZED)
                msg.data = "NOT_RELOCALIZED";
            else if (reloc_stat == RELOCALIZED)
                msg.data = myprintf("RELOCALIZED. [%7.2f, %7.2f, %7.2f, %7.2f, %7.2f, %7.2f]",
                                    tf_Lprior_L0.pos(0), tf_Lprior_L0.pos(1),  tf_Lprior_L0.pos(2),
                                    tf_Lprior_L0.yaw(),  tf_Lprior_L0.pitch(), tf_Lprior_L0.roll());
            else if (reloc_stat == RELOCALIZING)
                msg.data = "RELOCALIZING";
            reloc_pose_pub.publish(msg);
        }
    }

    void SlideWindowForward()
    {
        // Pop the states and replicate the final state
        ssQua.pop_front(); ssQua.push_back(deque<Quaternd>(N_SUB_SEG, sfQua.back().back()));
        ssPos.pop_front(); ssPos.push_back(deque<Vector3d>(N_SUB_SEG, sfPos.back().back()));
        ssVel.pop_front(); ssVel.push_back(deque<Vector3d>(N_SUB_SEG, sfVel.back().back()));
        ssBia.pop_front(); ssBia.push_back(deque<Vector3d>(N_SUB_SEG, sfBia.back().back()));
        ssBig.pop_front(); ssBig.push_back(deque<Vector3d>(N_SUB_SEG, sfBig.back().back()));

        sfQua.pop_front(); sfQua.push_back(ssQua.back());
        sfPos.pop_front(); sfPos.push_back(ssPos.back());
        sfVel.pop_front(); sfVel.push_back(ssVel.back());
        sfBia.pop_front(); sfBia.push_back(ssBia.back());
        sfBig.pop_front(); sfBig.push_back(ssBig.back());

        // Pop the buffers
        SwTimeStep.pop_front();
        SwCloud.pop_front();
        SwCloudDsk.pop_front();
        SwCloudDskDS.pop_front();
        SwLidarCoef.pop_front();
        SwDepVsAssoc.pop_front();
        SwImuBundle.pop_front();
        SwPropState.pop_front();
    }

    bool PublishGlobalMaps(slict::globalMapsPublish::Request &req, slict::globalMapsPublish::Response &res)
    {
        // Log and save the trajectory
        SaveTrajLog();

        // Publish the full map
        Util::publishCloud(global_map_pub, *globalMap, ros::Time(KfCloudPose->points.back().t), current_ref_frame);

        res.result = 1;
        return true;
    }

    void SaveTrajLog()
    {
        printf("Logging cloud pose: %s.\n", (log_dir + "/KfCloudPose.pcd").c_str());
        
        int save_attempts = 0;
        int save_attempts_max = 50;
        CloudPose cloudTemp;
        PCDWriter writer;

        save_attempts = 0;
        writer.write<PointPose>(log_dir + "/KfCloudPoseBin.pcd", *KfCloudPose, 18);
        while (true)
        {
            writer.write(log_dir + "/KfCloudPoseBin.pcd", *KfCloudPose, 18);
            save_attempts++;

            bool saving_succeeded = false;
            if(pcl::io::loadPCDFile<PointPose>(log_dir + "/KfCloudPoseBin.pcd", cloudTemp) == 0)
            {
                if (cloudTemp.size() == KfCloudPose->size())
                    saving_succeeded = true;
            }
            
            if (saving_succeeded)
                break;
            else if (save_attempts > save_attempts_max)
            {
                printf(KRED "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d!. Giving up \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
                break;

            }
            else
            {
                printf(KYEL "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
            }
        }

        save_attempts = 0;
        writer.writeASCII<PointPose>(log_dir + "/KfCloudPose.pcd", *KfCloudPose, 18);
        while (true)
        {
            writer.writeASCII<PointPose>(log_dir + "/KfCloudPose.pcd", *KfCloudPose, 18);
            save_attempts++;

            bool saving_succeeded = false;
            if(pcl::io::loadPCDFile<PointPose>(log_dir + "/KfCloudPose.pcd", cloudTemp) == 0)
            {
                if (cloudTemp.size() == KfCloudPose->size())
                    saving_succeeded = true;
            }
            
            if (saving_succeeded)
                break;
            else if (save_attempts > save_attempts_max)
            {
                printf(KRED "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d!. Giving up \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
                break;

            }
            else
            {
                printf(KYEL "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
            }
        }

        save_attempts = 0;
        writer.writeASCII<PointPose>(log_dir + "/KfCloudPoseExtra.pcd", *KfCloudPose, 18);
        while (true)
        {
            writer.writeASCII<PointPose>(log_dir + "/KfCloudPoseExtra.pcd", *KfCloudPose, 18);
            save_attempts++;

            bool saving_succeeded = false;
            if(pcl::io::loadPCDFile<PointPose>(log_dir + "/KfCloudPoseExtra.pcd", cloudTemp) == 0)
            {
                if (cloudTemp.size() == KfCloudPose->size())
                    saving_succeeded = true;
            }
            
            if (saving_succeeded)
                break;
            else if (save_attempts > save_attempts_max)
            {
                printf(KRED "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d!. Giving up \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
                break;

            }
            else
            {
                printf(KYEL "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
            }
        }

        printf(KGRN "Logging the map completed.\n" RESET);


        printf(KYEL "Logging the map start ...\n" RESET);

        {
            lock_guard<mutex> lock(global_map_mtx);

            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setRadiusSearch(max(leaf_size, 0.2));
            downsampler.setInputCloud(globalMap);
            downsampler.filter(*globalMap);

            printf("Logging global map: %s.\n", (log_dir + "/globalMap.pcd").c_str());
            pcl::io::savePCDFileBinary(log_dir + "/globalMap.pcd", *globalMap);

            #pragma omp parallel for num_threads(MAX_THREADS)
            for(int i = 0; i < KfCloudinW.size(); i++)
            {
                string file_name = log_dir_kf + "/KfCloudinW_" + zeroPaddedString(i, KfCloudinW.size()) + ".pcd";
                
                // printf("Logging KF cloud %s.\n", file_name.c_str());
                pcl::io::savePCDFileBinary(file_name, *KfCloudinW[i]);
            }
        }

        printf(KGRN "Logging the map completed.\n" RESET);


        printf(KYEL "Logging the loop ...\n" RESET);

        std::ofstream loop_log_file;
        loop_log_file.open(log_dir + "/loop_log.csv");
        loop_log_file.precision(std::numeric_limits<double>::digits10 + 1);

        for(auto &loop : loopPairs)
        {
            loop_log_file << loop.currPoseId << ", "
                          << loop.prevPoseId << ", "
                          << loop.JKavr << ", "
                          << loop.IcpFn << ", "
                          << loop.tf_Bp_Bc.pos(0) << ", "
                          << loop.tf_Bp_Bc.pos(1) << ", "
                          << loop.tf_Bp_Bc.pos(2) << ", "
                          << loop.tf_Bp_Bc.rot.x() << ", "
                          << loop.tf_Bp_Bc.rot.y() << ", "
                          << loop.tf_Bp_Bc.rot.z() << ", "
                          << loop.tf_Bp_Bc.rot.w() << endl;
        }

        loop_log_file.close();

        printf(KGRN "Logging the loop completed.\n" RESET);


        printf(KYEL "Logging the spline.\n" RESET);

        LogSpline(log_dir + "/spline_log.csv", *GlobalTraj, 0);

        printf(KYEL "Logging the spline completed.\n" RESET);

    }

    void LogSpline(string filename, PoseSplineX &traj, int outer_iteration)
    {
        std::ofstream spline_log_file;

        // Sample the spline from start to end
        spline_log_file.open(filename);
        spline_log_file.precision(std::numeric_limits<double>::digits10 + 1);

        // First row gives some metrics
        spline_log_file
                << "Dt: "         << traj.getDt()
                << ", Order: "    << SPLINE_N
                << ", Knots: "    << traj.numKnots()
                << ", MinTime: "  << traj.minTime()
                << ", MaxTime: "  << traj.maxTime()
                << ", OtrItr: "   << outer_iteration
                << endl;

        // Logging the knots
        for(int i = 0; i < traj.numKnots(); i++)
        {
            auto pose = traj.getKnot(i);
            auto pos = pose.translation(); auto rot = pose.so3().unit_quaternion();

            spline_log_file << i << ","
                            << traj.getKnotTime(i) << ","            
                            << pos.x() << "," << pos.y() << "," << pos.z() << ","
                            << rot.x() << "," << rot.y() << "," << rot.z() << "," << rot.w()
                            << endl;
        }

        spline_log_file.close();

    }

    void PublishOdom(ros::Publisher &pub, Vector3d &pos, Quaternd &qua,
                     Vector3d &vel, Vector3d &gyr, Vector3d &acc,
                     Vector3d &bg, Vector3d &ba, ros::Time stamp, string frame)
    {
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = frame;
        odom_msg.child_frame_id  = "body";

        odom_msg.pose.pose.position.x = pos.x();
        odom_msg.pose.pose.position.y = pos.y();
        odom_msg.pose.pose.position.z = pos.z();

        odom_msg.pose.pose.orientation.x = qua.x();
        odom_msg.pose.pose.orientation.y = qua.y();
        odom_msg.pose.pose.orientation.z = qua.z();
        odom_msg.pose.pose.orientation.w = qua.w();

        odom_msg.twist.twist.linear.x = vel.x();
        odom_msg.twist.twist.linear.y = vel.y();
        odom_msg.twist.twist.linear.z = vel.z();

        odom_msg.twist.twist.angular.x = gyr.x();
        odom_msg.twist.twist.angular.y = gyr.y();
        odom_msg.twist.twist.angular.z = gyr.z();

        odom_msg.twist.covariance[0] = acc.x();
        odom_msg.twist.covariance[1] = acc.y();
        odom_msg.twist.covariance[2] = acc.z();

        odom_msg.twist.covariance[3] = bg.x();
        odom_msg.twist.covariance[4] = bg.y();
        odom_msg.twist.covariance[5] = bg.z();
        odom_msg.twist.covariance[6] = ba.x();
        odom_msg.twist.covariance[7] = ba.y();
        odom_msg.twist.covariance[8] = ba.z();

        pub.publish(odom_msg);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Estimator");
    ros::NodeHandle nh("~");
    ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    ROS_INFO(KGRN "----> Estimator Started." RESET);

    Estimator estimator(nh_ptr);

    thread process_data(&Estimator::ProcessData, &estimator);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    estimator.SaveTrajLog();

    return 0;
}
