package org.mapreduce.utils;

import java.awt.image.BufferedImage;
import java.util.Map;

/**
 * Ray generation utility for FrugalNeRF
 * Generates rays for volume rendering
 */
public class RayGenerator {
    
    /**
     * Generate rays for given image and pose data
     */
    public float[][][] generateRays(BufferedImage image, Map<String, Object> poseData) {
        int width = image.getWidth();
        int height = image.getHeight();
        
        // Extract pose and intrinsics
        float[][] pose = (float[][]) poseData.get("pose");
        float[][] intrinsics = (float[][]) poseData.get("intrinsics");
        
        // Generate rays for each pixel
        float[][][] rays = new float[height][width][6]; // 6 values: 3 for origin, 3 for direction
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float[] ray = generateRayForPixel(x, y, width, height, pose, intrinsics);
                rays[y][x] = ray;
            }
        }
        
        return rays;
    }
    
    /**
     * Generate ray for a single pixel
     */
    private float[] generateRayForPixel(int x, int y, int width, int height, 
                                      float[][] pose, float[][] intrinsics) {
        // Convert pixel coordinates to normalized device coordinates
        float u = (float) x / width;
        float v = (float) y / height;
        
        // Convert to camera coordinates
        float fx = intrinsics[0][0];
        float fy = intrinsics[1][1];
        float cx = intrinsics[0][2];
        float cy = intrinsics[1][2];
        
        // Ray direction in camera coordinates
        float rayDirX = (x - cx) / fx;
        float rayDirY = (y - cy) / fy;
        float rayDirZ = 1.0f;
        
        // Normalize ray direction
        float length = (float) Math.sqrt(rayDirX * rayDirX + rayDirY * rayDirY + rayDirZ * rayDirZ);
        rayDirX /= length;
        rayDirY /= length;
        rayDirZ /= length;
        
        // Transform to world coordinates using pose
        float[] rayOrigin = new float[3];
        float[] rayDirection = new float[3];
        
        // Ray origin (camera position)
        rayOrigin[0] = pose[0][3];
        rayOrigin[1] = pose[1][3];
        rayOrigin[2] = pose[2][3];
        
        // Transform ray direction
        rayDirection[0] = pose[0][0] * rayDirX + pose[0][1] * rayDirY + pose[0][2] * rayDirZ;
        rayDirection[1] = pose[1][0] * rayDirX + pose[1][1] * rayDirY + pose[1][2] * rayDirZ;
        rayDirection[2] = pose[2][0] * rayDirX + pose[2][1] * rayDirY + pose[2][2] * rayDirZ;
        
        // Return ray as 6 values: [origin_x, origin_y, origin_z, direction_x, direction_y, direction_z]
        return new float[]{
            rayOrigin[0], rayOrigin[1], rayOrigin[2],
            rayDirection[0], rayDirection[1], rayDirection[2]
        };
    }
    
    /**
     * Generate rays for multiple views
     */
    public float[][][][] generateRaysForViews(BufferedImage image, Map<String, Object>[] viewPoses) {
        int width = image.getWidth();
        int height = image.getHeight();
        int numViews = viewPoses.length;
        
        // Generate rays for each view
        float[][][][] allRays = new float[numViews][height][width][6];
        
        for (int v = 0; v < numViews; v++) {
            allRays[v] = generateRays(image, viewPoses[v]);
        }
        
        return allRays;
    }
    
    /**
     * Calculate ray intersection with bounding box
     */
    public float[] calculateRayBounds(float[] rayOrigin, float[] rayDirection, 
                                    float[] bboxMin, float[] bboxMax) {
        float tMin = Float.NEGATIVE_INFINITY;
        float tMax = Float.POSITIVE_INFINITY;
        
        for (int i = 0; i < 3; i++) {
            if (Math.abs(rayDirection[i]) < 1e-8) {
                // Ray is parallel to the plane
                if (rayOrigin[i] < bboxMin[i] || rayOrigin[i] > bboxMax[i]) {
                    return null; // Ray doesn't intersect
                }
            } else {
                float t1 = (bboxMin[i] - rayOrigin[i]) / rayDirection[i];
                float t2 = (bboxMax[i] - rayOrigin[i]) / rayDirection[i];
                
                if (t1 > t2) {
                    float temp = t1;
                    t1 = t2;
                    t2 = temp;
                }
                
                tMin = Math.max(tMin, t1);
                tMax = Math.min(tMax, t2);
            }
        }
        
        if (tMin > tMax || tMax < 0) {
            return null; // Ray doesn't intersect
        }
        
        return new float[]{Math.max(0, tMin), tMax};
    }
}
