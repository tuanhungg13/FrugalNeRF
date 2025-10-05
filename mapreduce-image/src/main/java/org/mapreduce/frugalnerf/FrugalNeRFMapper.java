package org.mapreduce.frugalnerf;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.mapreduce.utils.ImageUtils;
import org.mapreduce.utils.DepthEstimator;
import org.mapreduce.utils.RayGenerator;
import org.mapreduce.utils.PoseProcessor;
import org.mapreduce.utils.MetadataParser;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

/**
 * Mapper cho FrugalNeRF data preprocessing
 * Xử lý ảnh, estimate depth, generate rays
 */
public class FrugalNeRFMapper extends Mapper<Text, BytesWritable, Text, BytesWritable> {

    // Configuration parameters
    private static final int TARGET_W = 256;
    private static final int TARGET_H = 256;
    private static final boolean ESTIMATE_DEPTH = true;
    private static final boolean GENERATE_RAYS = true;
    
    private DepthEstimator depthEstimator;
    private RayGenerator rayGenerator;
    private PoseProcessor poseProcessor;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);
        
        // Initialize processors
        this.depthEstimator = new DepthEstimator();
        this.rayGenerator = new RayGenerator();
        this.poseProcessor = new PoseProcessor();
        
        // Get configuration parameters
        int targetW = context.getConfiguration().getInt("frugalnerf.target.width", TARGET_W);
        int targetH = context.getConfiguration().getInt("frugalnerf.target.height", TARGET_H);
        boolean estimateDepth = context.getConfiguration().getBoolean("frugalnerf.estimate.depth", ESTIMATE_DEPTH);
        boolean generateRays = context.getConfiguration().getBoolean("frugalnerf.generate.rays", GENERATE_RAYS);
        
        System.out.println("[Mapper:setup] target=" + targetW + "x" + targetH + ", estimateDepth=" + estimateDepth + ", generateRays=" + generateRays);
        context.getCounter("SETUP", "INITIALIZED").increment(1);
    }

    @Override
    protected void map(Text key, BytesWritable value, Context context) 
            throws IOException, InterruptedException {
        
        String filename = key.toString();
        byte[] imgBytes = value.copyBytes();

        try {
            System.out.println("[Mapper:input] key=" + filename + ", valueBytes=" + imgBytes.length);
            
            // 1. Load and preprocess image
            BufferedImage img = ImageUtils.bytesToBufferedImage(imgBytes);
            if (img == null) {
                context.getCounter("ERRORS", "NULL_IMAGE").increment(1);
                System.err.println("Failed to load image: " + filename);
                return;
            }
            System.out.println("[Mapper:image] loaded W=" + img.getWidth() + ", H=" + img.getHeight());

            // Resize image
            BufferedImage resized = ImageUtils.resize(img, TARGET_W, TARGET_H);
            System.out.println("[Mapper:image] resized to " + TARGET_W + "x" + TARGET_H);
            
            // 2. Estimate depth (if enabled) - with error handling
            float[][] depthMap = null;
            if (ESTIMATE_DEPTH) {
                try {
                    depthMap = depthEstimator.estimateDepth(resized);
                    context.getCounter("PROCESSED", "DEPTH_ESTIMATED").increment(1);
                    System.out.println("[Mapper:depth] estimated: H=" + depthMap.length + ", W=" + (depthMap.length>0?depthMap[0].length:0));
                } catch (Exception e) {
                    System.err.println("[Mapper:depth] failed: " + e.getMessage());
                    // Continue without depth
                }
            }

            // 3. Generate rays (if enabled) - with error handling
            float[][][] rays = null;
            if (GENERATE_RAYS) {
                try {
                    // Extract pose and intrinsics from filename or metadata
                    Map<String, Object> poseData = extractPoseData(filename);
                    rays = rayGenerator.generateRays(resized, poseData);
                    context.getCounter("PROCESSED", "RAYS_GENERATED").increment(1);
                    System.out.println("[Mapper:rays] generated");
                } catch (Exception e) {
                    System.err.println("[Mapper:rays] failed: " + e.getMessage());
                    // Continue without rays
                }
            }

            // 4. Create processed data structure
            ProcessedImageData processedData = new ProcessedImageData(
                filename,
                resized,
                depthMap,
                rays
            );
            System.out.println("[Mapper:output] processed struct built");

            // 5. Serialize and emit - with error handling
            try {
                byte[] serializedData = serializeProcessedData(processedData);
                String sceneKey = extractSceneId(filename);
                System.out.println("[Mapper:emit] key=" + sceneKey + ", valueBytes=" + serializedData.length);
                context.write(new Text(sceneKey), new BytesWritable(serializedData));
                context.getCounter("PROCESSED", "IMAGES").increment(1);
                System.out.println("[Mapper:done] " + filename);
            } catch (Exception e) {
                System.err.println("[Mapper:serialize] failed: " + e.getMessage());
                // Emit simple data as fallback
                String sceneKey = extractSceneId(filename);
                System.out.println("[Mapper:emit-fallback] key=" + sceneKey + ", valueBytes=" + imgBytes.length);
                context.write(new Text(sceneKey), new BytesWritable(imgBytes));
                context.getCounter("PROCESSED", "IMAGES").increment(1);
                System.out.println("[Mapper:done-fallback] " + filename);
            }
            
        } catch (Exception e) {
            context.getCounter("ERRORS", "PROCESSING_FAILED").increment(1);
            System.err.println("[Mapper:error] critical: " + e.getMessage());
            e.printStackTrace();
            
            // Emit original data as fallback
            try {
                String sceneKey = extractSceneId(filename);
                System.out.println("[Mapper:emit-original] key=" + sceneKey + ", valueBytes=" + imgBytes.length);
                context.write(new Text(sceneKey), new BytesWritable(imgBytes));
                context.getCounter("PROCESSED", "IMAGES").increment(1);
                System.out.println("[Mapper:done-original] " + filename);
            } catch (Exception e2) {
                System.err.println("[Mapper:emit-original] failed: " + e2.getMessage());
            }
        }
    }

    /**
     * Extract pose and intrinsics data from minimal input (only images + poses_bounds.npy)
     */
    private Map<String, Object> extractPoseData(String filename) {
        Map<String, Object> poseData = new HashMap<>();
        
        try {
            // Extract scene directory from filename
            String sceneDir = extractSceneDirectory(filename);
            
            // Extract image index from filename
            int imageIndex = extractImageIndex(filename);
            poseData.put("image_index", imageIndex);
            
            // Try to parse poses_bounds.npy (main source of truth)
            String posesPath = sceneDir + "/poses_bounds.npy";
            Map<String, Object> posesData = MetadataParser.parsePosesBounds(posesPath);
            
            if (posesData.containsKey("poses") && posesData.containsKey("bounds")) {
                // Use poses_bounds.npy data
                float[][][] allPoses = (float[][][]) posesData.get("poses");
                float[][] allBounds = (float[][]) posesData.get("bounds");
                
                if (imageIndex < allPoses.length) {
                    poseData.put("pose", allPoses[imageIndex]);
                    poseData.put("bounds", allBounds[imageIndex]);
                } else {
                    // Use last available pose if index out of bounds
                    int lastIndex = allPoses.length - 1;
                    poseData.put("pose", allPoses[lastIndex]);
                    poseData.put("bounds", allBounds[lastIndex]);
                }
                
                System.out.println("Successfully loaded pose data from poses_bounds.npy for image " + imageIndex);
            } else {
                throw new Exception("poses_bounds.npy not found or invalid");
            }
            
            // Try to parse intrinsics (optional)
            String intrinsicsPath = sceneDir + "/intrinsics.npy";
            try {
                Map<String, Object> intrinsicsData = MetadataParser.parseIntrinsics(intrinsicsPath);
                if (intrinsicsData.containsKey("intrinsics")) {
                    poseData.put("intrinsics", intrinsicsData.get("intrinsics"));
                    System.out.println("Loaded intrinsics from intrinsics.npy");
                } else {
                    throw new Exception("intrinsics.npy not found");
                }
            } catch (Exception e) {
                // Generate intrinsics from image dimensions
                poseData.put("intrinsics", generateIntrinsicsFromImage(filename));
                System.out.println("Generated intrinsics from image dimensions");
            }
            
            // Try to parse scene metadata (optional)
            String metadataPath = sceneDir + "/metadata.json";
            try {
                Map<String, Object> metadata = MetadataParser.parseSceneMetadata(metadataPath);
                poseData.putAll(metadata);
                System.out.println("Loaded metadata from metadata.json");
            } catch (Exception e) {
                // Generate default metadata
                poseData.put("scene_bbox", generateDefaultSceneBbox());
                poseData.put("near_far", new float[]{0.1f, 10.0f});
                poseData.put("white_bg", false);
                System.out.println("Using default scene metadata");
            }
            
        } catch (Exception e) {
            System.err.println("Error extracting pose data for " + filename + ": " + e.getMessage());
            System.err.println("Falling back to default values");
            
            // Fallback to default values
            poseData.put("pose", new float[][]{
                {1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}
            });
            poseData.put("intrinsics", generateIntrinsicsFromImage(filename));
            poseData.put("bounds", new float[]{0.1f, 10.0f});
            poseData.put("scene_bbox", generateDefaultSceneBbox());
            poseData.put("near_far", new float[]{0.1f, 10.0f});
            poseData.put("white_bg", false);
            poseData.put("image_index", 0);
        }
        
        return poseData;
    }
    
    /**
     * Extract scene directory from filename
     */
    private String extractSceneDirectory(String filename) {
        // Extract scene directory from path like "/frugalnerf/input/scene_001/images/image_001.jpg"
        String[] parts = filename.split("/");
        StringBuilder sceneDir = new StringBuilder();
        
        for (int i = 0; i < parts.length - 2; i++) { // Remove "images" and filename
            if (i > 0) sceneDir.append("/");
            sceneDir.append(parts[i]);
        }
        
        return sceneDir.toString();
    }
    
    /**
     * Extract image index from filename
     */
    private int extractImageIndex(String filename) {
        // Extract from filename like "image_001.jpg" -> 1
        String[] parts = filename.split("_");
        if (parts.length >= 2) {
            String indexStr = parts[1].split("\\.")[0];
            try {
                return Integer.parseInt(indexStr);
            } catch (NumberFormatException e) {
                return 0;
            }
        }
        return 0;
    }
    
    /**
     * Generate intrinsics from image dimensions
     */
    private float[][] generateIntrinsicsFromImage(String filename) {
        // Default intrinsics based on common image sizes
        // Assume 256x256 images with focal length ~500
        return new float[][]{
            {500.0f, 0.0f, 128.0f},  // fx, 0, cx
            {0.0f, 500.0f, 128.0f},  // 0, fy, cy  
            {0.0f, 0.0f, 1.0f}        // 0, 0, 1
        };
    }
    
    /**
     * Generate default scene bounding box
     */
    private float[][] generateDefaultSceneBbox() {
        return new float[][]{
            {-1.5f, -1.5f, -1.5f},  // min
            {1.5f, 1.5f, 1.5f}       // max
        };
    }
    
    /**
     * Extract scene ID from filename
     */
    private String extractSceneId(String filename) {
        // Extract scene ID from path like "/frugalnerf/input/scene_001/images/image_001.jpg"
        String[] parts = filename.split("/");
        for (String part : parts) {
            if (part.startsWith("scene_")) {
                return part;
            }
        }
        return "scene_001"; // Default fallback
    }
    
    /**
     * Load poses from numpy file
     */
    private float[][] loadPosesFromNumpy(String posesPath) {
        // In real implementation, load from .npy file
        // For now, return default poses
        return new float[][]{
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
        };
    }
    
    /**
     * Load intrinsics from numpy file
     */
    private float[][] loadIntrinsicsFromNumpy(String intrinsicsPath) {
        // In real implementation, load from .npy file
        return new float[][]{
            {500.0f, 0.0f, 256.0f},
            {0.0f, 500.0f, 256.0f},
            {0.0f, 0.0f, 1.0f}
        };
    }
    
    /**
     * Load metadata from JSON file
     */
    private Map<String, Object> loadMetadata(String metadataPath) {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("scene_bbox", new float[][]{
            {-1.5f, -1.5f, -1.5f},
            {1.5f, 1.5f, 1.5f}
        });
        metadata.put("near_far", new float[]{0.1f, 10.0f});
        metadata.put("white_bg", false);
        return metadata;
    }

    /**
     * Serialize processed data to bytes
     */
    private byte[] serializeProcessedData(ProcessedImageData data) {
        // Simple serialization - in production, use more efficient format
        StringBuilder sb = new StringBuilder();
        sb.append("FILENAME:").append(data.filename).append("\n");
        sb.append("WIDTH:").append(data.width).append("\n");
        sb.append("HEIGHT:").append(data.height).append("\n");
        
        if (data.depthMap != null) {
            sb.append("DEPTH_MAP:");
            for (float[] row : data.depthMap) {
                for (float val : row) {
                    sb.append(val).append(",");
                }
            }
            sb.append("\n");
        }
        
        if (data.rays != null) {
            sb.append("RAYS:");
            for (float[][] rayRow : data.rays) {
                for (float[] ray : rayRow) {
                    for (float val : ray) {
                        sb.append(val).append(",");
                    }
                }
            }
            sb.append("\n");
        }
        
        return sb.toString().getBytes();
    }

    /**
     * Data structure for processed image
     */
    private static class ProcessedImageData {
        String filename;
        BufferedImage image;
        float[][] depthMap;
        float[][][] rays;
        int width, height;

        public ProcessedImageData(String filename, BufferedImage image, 
                                float[][] depthMap, float[][][] rays) {
            this.filename = filename;
            this.image = image;
            this.depthMap = depthMap;
            this.rays = rays;
            this.width = image.getWidth();
            this.height = image.getHeight();
        }
    }
}
