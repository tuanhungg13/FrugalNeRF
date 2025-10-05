package org.mapreduce.utils;

import java.util.Map;
import java.util.HashMap;

/**
 * Pose processing utility for FrugalNeRF
 * Handles pose and intrinsics data
 */
public class PoseProcessor {
    
    /**
     * Parse pose data from filename or metadata
     */
    public Map<String, Object> parsePoseData(String filename) {
        Map<String, Object> poseData = new HashMap<>();
        
        // Extract scene ID and image ID from filename
        // Example: "scene_001_image_002.jpg" -> scene_001, image_002
        String[] parts = filename.split("_");
        if (parts.length >= 3) {
            String sceneId = parts[0] + "_" + parts[1];
            String imageId = parts[2].split("\\.")[0];
            
            poseData.put("scene_id", sceneId);
            poseData.put("image_id", imageId);
        }
        
        // Generate default pose (identity matrix)
        float[][] pose = generateDefaultPose();
        poseData.put("pose", pose);
        
        // Generate default intrinsics
        float[][] intrinsics = generateDefaultIntrinsics();
        poseData.put("intrinsics", intrinsics);
        
        return poseData;
    }
    
    /**
     * Generate default pose (identity matrix)
     */
    private float[][] generateDefaultPose() {
        return new float[][]{
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
        };
    }
    
    /**
     * Generate default camera intrinsics
     */
    private float[][] generateDefaultIntrinsics() {
        return new float[][]{
            {500.0f, 0.0f, 128.0f},
            {0.0f, 500.0f, 128.0f},
            {0.0f, 0.0f, 1.0f}
        };
    }
    
    /**
     * Validate pose matrix
     */
    public boolean validatePose(float[][] pose) {
        if (pose == null || pose.length != 4 || pose[0].length != 4) {
            return false;
        }
        
        // Check if it's a valid transformation matrix
        // Last row should be [0, 0, 0, 1]
        if (pose[3][0] != 0.0f || pose[3][1] != 0.0f || 
            pose[3][2] != 0.0f || pose[3][3] != 1.0f) {
            return false;
        }
        
        return true;
    }
    
    /**
     * Validate intrinsics matrix
     */
    public boolean validateIntrinsics(float[][] intrinsics) {
        if (intrinsics == null || intrinsics.length != 3 || intrinsics[0].length != 3) {
            return false;
        }
        
        // Check if it's a valid camera matrix
        // fx and fy should be positive
        if (intrinsics[0][0] <= 0 || intrinsics[1][1] <= 0) {
            return false;
        }
        
        // Principal point should be reasonable
        if (intrinsics[0][2] < 0 || intrinsics[1][2] < 0) {
            return false;
        }
        
        return true;
    }
    
    /**
     * Convert pose to different formats
     */
    public Map<String, Object> convertPoseFormat(float[][] pose, String format) {
        Map<String, Object> result = new HashMap<>();
        
        switch (format.toLowerCase()) {
            case "quaternion":
                result.put("quaternion", poseToQuaternion(pose));
                break;
            case "euler":
                result.put("euler", poseToEuler(pose));
                break;
            case "axis_angle":
                result.put("axis_angle", poseToAxisAngle(pose));
                break;
            default:
                result.put("matrix", pose);
        }
        
        return result;
    }
    
    /**
     * Convert pose matrix to quaternion
     */
    private float[] poseToQuaternion(float[][] pose) {
        // Extract rotation matrix
        float[][] R = new float[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R[i][j] = pose[i][j];
            }
        }
        
        // Convert rotation matrix to quaternion
        float[] quat = new float[4];
        float trace = R[0][0] + R[1][1] + R[2][2];
        
        if (trace > 0) {
            float s = (float) Math.sqrt(trace + 1.0f) * 2;
            quat[0] = (R[2][1] - R[1][2]) / s;
            quat[1] = (R[0][2] - R[2][0]) / s;
            quat[2] = (R[1][0] - R[0][1]) / s;
            quat[3] = 0.25f * s;
        } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
            float s = (float) Math.sqrt(1.0f + R[0][0] - R[1][1] - R[2][2]) * 2;
            quat[0] = 0.25f * s;
            quat[1] = (R[0][1] + R[1][0]) / s;
            quat[2] = (R[0][2] + R[2][0]) / s;
            quat[3] = (R[2][1] - R[1][2]) / s;
        } else if (R[1][1] > R[2][2]) {
            float s = (float) Math.sqrt(1.0f + R[1][1] - R[0][0] - R[2][2]) * 2;
            quat[0] = (R[0][1] + R[1][0]) / s;
            quat[1] = 0.25f * s;
            quat[2] = (R[1][2] + R[2][1]) / s;
            quat[3] = (R[0][2] - R[2][0]) / s;
        } else {
            float s = (float) Math.sqrt(1.0f + R[2][2] - R[0][0] - R[1][1]) * 2;
            quat[0] = (R[0][2] + R[2][0]) / s;
            quat[1] = (R[1][2] + R[2][1]) / s;
            quat[2] = 0.25f * s;
            quat[3] = (R[1][0] - R[0][1]) / s;
        }
        
        return quat;
    }
    
    /**
     * Convert pose matrix to Euler angles
     */
    private float[] poseToEuler(float[][] pose) {
        float[][] R = new float[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R[i][j] = pose[i][j];
            }
        }
        
        float[] euler = new float[3];
        
        // Extract Euler angles (ZYX convention)
        euler[0] = (float) Math.atan2(R[2][1], R[2][2]);
        euler[1] = (float) Math.asin(-R[2][0]);
        euler[2] = (float) Math.atan2(R[1][0], R[0][0]);
        
        return euler;
    }
    
    /**
     * Convert pose matrix to axis-angle representation
     */
    private float[] poseToAxisAngle(float[][] pose) {
        float[] quat = poseToQuaternion(pose);
        
        float[] axisAngle = new float[4];
        float angle = 2 * (float) Math.acos(quat[3]);
        
        if (angle > 0) {
            axisAngle[0] = quat[0] / (float) Math.sin(angle / 2);
            axisAngle[1] = quat[1] / (float) Math.sin(angle / 2);
            axisAngle[2] = quat[2] / (float) Math.sin(angle / 2);
        } else {
            axisAngle[0] = 0;
            axisAngle[1] = 0;
            axisAngle[2] = 0;
        }
        axisAngle[3] = angle;
        
        return axisAngle;
    }
}
