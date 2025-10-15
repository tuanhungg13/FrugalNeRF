package org.mapreduce.frugalnerf;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Reducer cho FrugalNeRF data preprocessing
 * Tổng hợp dữ liệu đã xử lý theo scene
 */
public class FrugalNeRFReducer extends Reducer<Text, BytesWritable, Text, BytesWritable> {

    @Override
    protected void reduce(Text key, Iterable<BytesWritable> values, Context context)
            throws IOException, InterruptedException {
        
        try {
            System.out.println("[Reducer:input] key=" + key.toString());
            // Collect all processed images for this scene (with dedup by filename + b64 if present)
            List<ProcessedImageData> processedImages = new ArrayList<>();
            java.util.HashSet<String> seen = new java.util.HashSet<>();
            
            for (BytesWritable value : values) {
                System.out.println("[Reducer:recv] valueBytes=" + value.getLength());
                ProcessedImageData imageData = deserializeProcessedData(value.copyBytes());
                if (imageData != null) {
                    String sig = (imageData.filename != null ? imageData.filename : "") + "|" + (imageData.imageB64 != null ? imageData.imageB64.hashCode() : 0);
                    if (seen.add(sig)) processedImages.add(imageData);
                }
            }
            
            if (processedImages.isEmpty()) {
                context.getCounter("ERRORS", "NO_VALID_IMAGES").increment(1);
                return;
            }
            
            // Create scene summary
            SceneData sceneData = createSceneData(key.toString(), processedImages);
            
            // Serialize scene data
            byte[] serializedScene = serializeSceneData(sceneData);
            
            // Emit scene data
            System.out.println("[Reducer:emit] key=scene_" + key.toString() + ", valueBytes=" + serializedScene.length);
            context.write(new Text("scene_" + key.toString()), new BytesWritable(serializedScene));
            
            // Update counters
            context.getCounter("SCENES", "PROCESSED").increment(1);
            context.getCounter("IMAGES", "TOTAL").increment(processedImages.size());
            
        } catch (Exception e) {
            context.getCounter("ERRORS", "REDUCER_EXCEPTION").increment(1);
            System.err.println("Error in reducer: " + e.getMessage());
        }
    }

    /**
     * Deserialize processed image data from bytes
     */
    private ProcessedImageData deserializeProcessedData(byte[] data) {
        try {
            String dataStr = new String(data);
            String[] lines = dataStr.split("\n");
            
            String filename = null;
            int width = 0, height = 0;
            float[][] depthMap = null;
            float[][][] rays = null;
            String imageB64 = null;
            
            for (String line : lines) {
                if (line.startsWith("FILENAME:")) {
                    filename = line.substring(9);
                } else if (line.startsWith("WIDTH:")) {
                    width = Integer.parseInt(line.substring(6));
                } else if (line.startsWith("HEIGHT:")) {
                    height = Integer.parseInt(line.substring(7));
                } else if (line.startsWith("IMAGE_BYTES_BASE64:")) {
                    imageB64 = line.substring(20);
                } else if (line.startsWith("DEPTH_MAP:")) {
                    depthMap = parseDepthMap(line.substring(10), width, height);
                } else if (line.startsWith("RAYS:")) {
                    rays = parseRays(line.substring(5), width, height);
                }
            }
            
            return new ProcessedImageData(filename, width, height, depthMap, rays, imageB64);
            
        } catch (Exception e) {
            System.err.println("Error deserializing data: " + e.getMessage());
            return null;
        }
    }

    /**
     * Parse depth map from string
     */
    private float[][] parseDepthMap(String data, int width, int height) {
        String[] values = data.split(",");
        float[][] depthMap = new float[height][width];
        
        int idx = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (idx < values.length && !values[idx].isEmpty()) {
                    depthMap[i][j] = Float.parseFloat(values[idx]);
                }
                idx++;
            }
        }
        
        return depthMap;
    }

    /**
     * Parse rays from string
     */
    private float[][][] parseRays(String data, int width, int height) {
        String[] values = data.split(",");
        float[][][] rays = new float[height][width][6]; // 6 values per ray (3 for origin, 3 for direction)
        
        int idx = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < 6; k++) {
                    if (idx < values.length && !values[idx].isEmpty()) {
                        rays[i][j][k] = Float.parseFloat(values[idx]);
                    }
                    idx++;
                }
            }
        }
        
        return rays;
    }

    /**
     * Create scene data from processed images
     */
    private SceneData createSceneData(String sceneId, List<ProcessedImageData> images) {
        SceneData sceneData = new SceneData();
        sceneData.sceneId = sceneId;
        sceneData.numImages = images.size();
        sceneData.images = images;
        
        // Calculate scene bounding box
        sceneData.sceneBbox = calculateSceneBbox(images);
        
        // Calculate near/far planes
        sceneData.nearFar = calculateNearFar(images);
        
        // Extract poses and intrinsics
        sceneData.poses = extractPoses(images);
        
        // Initialize bounds per image using near/far to avoid NPE during serialization
        sceneData.bounds = new float[images.size()][2];
        for (int i = 0; i < images.size(); i++) {
            sceneData.bounds[i][0] = sceneData.nearFar[0];
            sceneData.bounds[i][1] = sceneData.nearFar[1];
        }
        sceneData.intrinsics = extractIntrinsics(images);
        
        // Generate ray directions
        sceneData.directions = generateRayDirections(images);
        
        return sceneData;
    }
    
    /**
     * Extract poses from processed images
     */
    private float[][][] extractPoses(List<ProcessedImageData> images) {
        // Standardize to 3x5 pose blocks expected by serialization (3 rows x 5 cols)
        // Construct from a 4x4 camera matrix: take first 3x4 and set last column to [H, W, focal]
        float[][][] poses3x5 = new float[images.size()][3][5];
        for (int i = 0; i < images.size(); i++) {
            ProcessedImageData img = images.get(i);
            float[][] pose4x4 = extractPoseFromImage(img);
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 4; c++) {
                    poses3x5[i][r][c] = pose4x4[r][c];
                }
            }
            int width = img.width;
            int height = img.height;
            if (width <= 0 || height <= 0) {
                width = 256;
                height = 256;
            }
            float[][] intr = extractIntrinsicsFromImage(img);
            float focal = (intr != null) ? intr[0][0] : 0.0f;
            if (!Float.isFinite(focal) || focal <= 0.0f) {
                focal = 0.5f * (float) Math.min(width, height);
            }
            poses3x5[i][0][4] = height; // H
            poses3x5[i][1][4] = width;  // W
            poses3x5[i][2][4] = focal;  // focal
        }
        return poses3x5;
    }
    
    /**
     * Extract intrinsics from processed images
     */
    private float[][][] extractIntrinsics(List<ProcessedImageData> images) {
        float[][][] intrinsics = new float[images.size()][3][3];
        for (int i = 0; i < images.size(); i++) {
            // Extract intrinsics from image data
            intrinsics[i] = extractIntrinsicsFromImage(images.get(i));
        }
        return intrinsics;
    }
    
    /**
     * Generate ray directions for all images
     */
    private float[][][] generateRayDirections(List<ProcessedImageData> images) {
        if (images.isEmpty()) return new float[0][0][0];
        
        ProcessedImageData firstImage = images.get(0);
        int H = firstImage.height;
        int W = firstImage.width;
        
        // Generate ray directions (H, W, 3)
        float[][][] directions = new float[H][W][3];
        
        // Extract focal length from first image
        float[] focal = extractFocalFromImage(firstImage);
        
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                // Calculate ray direction
                float rayX = (x - W/2.0f) / focal[0];
                float rayY = (y - H/2.0f) / focal[1];
                float rayZ = 1.0f;
                
                // Normalize
                float length = (float) Math.sqrt(rayX*rayX + rayY*rayY + rayZ*rayZ);
                directions[y][x][0] = rayX / length;
                directions[y][x][1] = rayY / length;
                directions[y][x][2] = rayZ / length;
            }
        }
        
        return directions;
    }
    
    /**
     * Extract pose from image data
     */
    private float[][] extractPoseFromImage(ProcessedImageData image) {
        // Extract pose from image metadata
        // For now, return identity matrix
        return new float[][]{
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
        };
    }
    
    /**
     * Extract intrinsics from image data
     */
    private float[][] extractIntrinsicsFromImage(ProcessedImageData image) {
        // Extract intrinsics from image metadata
        return new float[][]{
            {500.0f, 0.0f, image.width/2.0f},
            {0.0f, 500.0f, image.height/2.0f},
            {0.0f, 0.0f, 1.0f}
        };
    }
    
    /**
     * Extract focal length from image data
     */
    private float[] extractFocalFromImage(ProcessedImageData image) {
        // Extract focal length from image metadata
        return new float[]{500.0f, 500.0f};
    }

    /**
     * Calculate scene bounding box from all images
     */
    private float[][] calculateSceneBbox(List<ProcessedImageData> images) {
        // Simple implementation - in production, use proper 3D bounding box calculation
        float minX = -1.5f, minY = -1.5f, minZ = -1.5f;
        float maxX = 1.5f, maxY = 1.5f, maxZ = 1.5f;
        
        return new float[][]{
            {minX, minY, minZ},
            {maxX, maxY, maxZ}
        };
    }

    /**
     * Calculate near/far planes
     */
    private float[] calculateNearFar(List<ProcessedImageData> images) {
        // Simple implementation
        return new float[]{0.1f, 10.0f};
    }

    /**
     * Serialize scene data to FrugalNeRF format
     */
    private byte[] serializeSceneData(SceneData sceneData) {
        StringBuilder sb = new StringBuilder();
        
        // Create FrugalNeRF compatible format
        sb.append("# FrugalNeRF Dataset Format\n");
        sb.append("# Generated by MapReduce preprocessing\n\n");
        
        // Scene metadata
        sb.append("SCENE_ID:").append(sceneData.sceneId).append("\n");
        sb.append("NUM_IMAGES:").append(sceneData.numImages).append("\n");
        sb.append("SCENE_BBOX:").append(sceneData.sceneBbox[0][0]).append(",")
          .append(sceneData.sceneBbox[0][1]).append(",").append(sceneData.sceneBbox[0][2]).append(",")
          .append(sceneData.sceneBbox[1][0]).append(",").append(sceneData.sceneBbox[1][1]).append(",")
          .append(sceneData.sceneBbox[1][2]).append("\n");
        sb.append("NEAR_FAR:").append(sceneData.nearFar[0]).append(",").append(sceneData.nearFar[1]).append("\n");
        sb.append("WHITE_BG:").append(sceneData.whiteBg).append("\n\n");
        
        // Poses in FrugalNeRF format (N x 17 array)
        sb.append("POSES_BOUNDS:\n");
        for (int i = 0; i < sceneData.poses.length; i++) {
            float[][] pose = sceneData.poses[i];
            float[] bounds = sceneData.bounds[i];
            
            // Flatten pose matrix (3x5 -> 15 values)
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 5; k++) {
                    sb.append(pose[j][k]).append(" ");
                }
            }
            // Add bounds (2 values)
            sb.append(bounds[0]).append(" ").append(bounds[1]).append("\n");
        }
        sb.append("\n");
        
        // Intrinsics (assume same for all images)
        sb.append("INTRINSICS:\n");
        float[][] intrinsic = sceneData.intrinsics[0]; // Use first image's intrinsics
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                sb.append(intrinsic[i][j]).append(" ");
            }
            sb.append("\n");
        }
        sb.append("\n");
        
        // Ray directions
        sb.append("DIRECTIONS:\n");
        for (float[][] directionRow : sceneData.directions) {
            for (float[] direction : directionRow) {
                for (float val : direction) {
                    sb.append(val).append(" ");
                }
                sb.append("\n");
            }
        }
        sb.append("\n");
        
        // Image data with depth and rays
        sb.append("IMAGES:\n");
        for (int i = 0; i < sceneData.images.size(); i++) {
            ProcessedImageData img = sceneData.images.get(i);
            sb.append("IMAGE_").append(i).append(":\n");
            sb.append("  FILENAME:").append(img.filename).append("\n");
            sb.append("  WIDTH:").append(img.width).append("\n");
            sb.append("  HEIGHT:").append(img.height).append("\n");
            
            if (img.depthMap != null) {
                sb.append("  DEPTH_MAP:\n");
                for (float[] row : img.depthMap) {
                    for (float val : row) {
                        sb.append(val).append(" ");
                    }
                    sb.append("\n");
                }
            }
            
            if (img.rays != null) {
                sb.append("  RAYS:\n");
                for (float[][] rayRow : img.rays) {
                    for (float[] ray : rayRow) {
                        for (float val : ray) {
                            sb.append(val).append(" ");
                        }
                        sb.append("\n");
                    }
                }
            }
            sb.append("\n");
        }
        
        return sb.toString().getBytes();
    }

    /**
     * Serialize depth map to string
     */
    private String serializeDepthMap(float[][] depthMap) {
        StringBuilder sb = new StringBuilder();
        for (float[] row : depthMap) {
            for (float val : row) {
                sb.append(val).append(",");
            }
        }
        return sb.toString();
    }

    /**
     * Serialize rays to string
     */
    private String serializeRays(float[][][] rays) {
        StringBuilder sb = new StringBuilder();
        for (float[][] rayRow : rays) {
            for (float[] ray : rayRow) {
                for (float val : ray) {
                    sb.append(val).append(",");
                }
            }
        }
        return sb.toString();
    }

    /**
     * Data structure for processed image
     */
    private static class ProcessedImageData {
        String filename;
        int width, height;
        float[][] depthMap;
        float[][][] rays;
        String imageB64;

        public ProcessedImageData(String filename, int width, int height, 
                                float[][] depthMap, float[][][] rays, String imageB64) {
            this.filename = filename;
            this.width = width;
            this.height = height;
            this.depthMap = depthMap;
            this.rays = rays;
            this.imageB64 = imageB64;
        }
    }

    /**
     * Data structure for scene
     */
    private static class SceneData {
        String sceneId;
        int numImages;
        List<ProcessedImageData> images;
        float[][] sceneBbox;
        float[] nearFar;
        float[][][] poses;        // Camera poses (N, 3, 5)
        float[][] bounds;         // Near/far bounds (N, 2)
        float[][][] intrinsics;  // Camera intrinsics (N, 3, 3)
        float[][][] directions;  // Ray directions (H, W, 3)
        boolean whiteBg;         // White background flag
    }
}
