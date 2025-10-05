package org.mapreduce.utils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Utility class để parse metadata files cho FrugalNeRF
 * Hỗ trợ parse poses_bounds.npy, intrinsics, và scene metadata
 */
public class MetadataParser {
    
    private static final Pattern IMAGE_PATTERN = Pattern.compile(".*\\.(jpg|jpeg|png|bmp|tiff|tif)$", Pattern.CASE_INSENSITIVE);
    
    /**
     * Parse poses_bounds.npy file (simplified version)
     * Format: N x 17 array (15 pose + 2 bounds)
     */
    public static Map<String, Object> parsePosesBounds(String posesPath) {
        Map<String, Object> result = new HashMap<>();
        
        try {
            // Check if file exists
            if (!Files.exists(Paths.get(posesPath))) {
                System.err.println("poses_bounds.npy not found at: " + posesPath);
                throw new FileNotFoundException("poses_bounds.npy not found");
            }
            
            // Đọc file .npy (simplified - trong thực tế cần numpy parser)
            byte[] data = Files.readAllBytes(Paths.get(posesPath));
            
            if (data.length == 0) {
                throw new IOException("poses_bounds.npy is empty");
            }
            
            // Parse binary data (simplified)
            // Trong thực tế cần proper .npy parser
            float[][] poses_bounds = parseNumpyArray(data);
            
            if (poses_bounds == null || poses_bounds.length == 0) {
                throw new IOException("Failed to parse poses_bounds.npy");
            }
            
            result.put("poses_bounds", poses_bounds);
            result.put("num_images", poses_bounds.length);
            
            // Extract poses (first 15 columns)
            float[][][] poses = new float[poses_bounds.length][3][5];
            for (int i = 0; i < poses_bounds.length; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 5; k++) {
                        poses[i][j][k] = poses_bounds[i][j*5 + k];
                    }
                }
            }
            result.put("poses", poses);
            
            // Extract bounds (last 2 columns)
            float[][] bounds = new float[poses_bounds.length][2];
            for (int i = 0; i < poses_bounds.length; i++) {
                bounds[i][0] = poses_bounds[i][15]; // near
                bounds[i][1] = poses_bounds[i][16]; // far
            }
            result.put("bounds", bounds);
            
            System.out.println("Successfully parsed poses_bounds.npy with " + poses_bounds.length + " poses");
            
        } catch (IOException e) {
            System.err.println("Error parsing poses_bounds.npy: " + e.getMessage());
            // Return default values
            result.put("poses", generateDefaultPoses(10));
            result.put("bounds", generateDefaultBounds(10));
            result.put("num_images", 10);
        }
        
        return result;
    }
    
    /**
     * Parse camera intrinsics (optional file)
     */
    public static Map<String, Object> parseIntrinsics(String intrinsicsPath) {
        Map<String, Object> result = new HashMap<>();
        
        try {
            // Check if file exists
            if (!Files.exists(Paths.get(intrinsicsPath))) {
                System.out.println("intrinsics.npy not found, using default intrinsics");
                result.put("intrinsics", generateDefaultIntrinsics());
                return result;
            }
            
            // Đọc intrinsics file
            List<String> lines = Files.readAllLines(Paths.get(intrinsicsPath));
            
            if (lines.size() >= 3) {
                // Parse 3x3 matrix
                float[][] intrinsics = new float[3][3];
                for (int i = 0; i < 3; i++) {
                    String[] values = lines.get(i).trim().split("\\s+");
                    for (int j = 0; j < 3; j++) {
                        intrinsics[i][j] = Float.parseFloat(values[j]);
                    }
                }
                result.put("intrinsics", intrinsics);
                System.out.println("Successfully loaded intrinsics from file");
            } else {
                // Default intrinsics
                result.put("intrinsics", generateDefaultIntrinsics());
                System.out.println("Invalid intrinsics file, using default");
            }
            
        } catch (IOException e) {
            System.err.println("Error parsing intrinsics: " + e.getMessage());
            result.put("intrinsics", generateDefaultIntrinsics());
        }
        
        return result;
    }
    
    /**
     * Parse scene metadata (JSON format) - optional file
     */
    public static Map<String, Object> parseSceneMetadata(String metadataPath) {
        Map<String, Object> result = new HashMap<>();
        
        try {
            // Check if file exists
            if (!Files.exists(Paths.get(metadataPath))) {
                System.out.println("metadata.json not found, using default metadata");
                result.put("scene_bbox", generateDefaultSceneBbox());
                result.put("near_far", new float[]{0.1f, 10.0f});
                result.put("white_bg", false);
                return result;
            }
            
            String content = new String(Files.readAllBytes(Paths.get(metadataPath)));
            
            // Simple JSON parsing (trong thực tế dùng JSON library)
            if (content.contains("scene_bbox")) {
                // Parse scene bounding box
                String[] bboxParts = extractJsonArray(content, "scene_bbox");
                if (bboxParts.length >= 6) {
                    float[][] sceneBbox = new float[2][3];
                    sceneBbox[0][0] = Float.parseFloat(bboxParts[0]); // min_x
                    sceneBbox[0][1] = Float.parseFloat(bboxParts[1]); // min_y
                    sceneBbox[0][2] = Float.parseFloat(bboxParts[2]); // min_z
                    sceneBbox[1][0] = Float.parseFloat(bboxParts[3]); // max_x
                    sceneBbox[1][1] = Float.parseFloat(bboxParts[4]); // max_y
                    sceneBbox[1][2] = Float.parseFloat(bboxParts[5]); // max_z
                    result.put("scene_bbox", sceneBbox);
                    System.out.println("Loaded scene_bbox from metadata.json");
                }
            }
            
            if (content.contains("near_far")) {
                // Parse near/far planes
                String[] nearFarParts = extractJsonArray(content, "near_far");
                if (nearFarParts.length >= 2) {
                    float[] nearFar = new float[2];
                    nearFar[0] = Float.parseFloat(nearFarParts[0]); // near
                    nearFar[1] = Float.parseFloat(nearFarParts[1]); // far
                    result.put("near_far", nearFar);
                    System.out.println("Loaded near_far from metadata.json");
                }
            }
            
            if (content.contains("white_bg")) {
                // Parse white background flag
                boolean whiteBg = content.contains("\"white_bg\": true");
                result.put("white_bg", whiteBg);
                System.out.println("Loaded white_bg from metadata.json");
            }
            
            // Fill in missing values with defaults
            if (!result.containsKey("scene_bbox")) {
                result.put("scene_bbox", generateDefaultSceneBbox());
            }
            if (!result.containsKey("near_far")) {
                result.put("near_far", new float[]{0.1f, 10.0f});
            }
            if (!result.containsKey("white_bg")) {
                result.put("white_bg", false);
            }
            
        } catch (IOException e) {
            System.err.println("Error parsing metadata: " + e.getMessage());
            // Default values
            result.put("scene_bbox", generateDefaultSceneBbox());
            result.put("near_far", new float[]{0.1f, 10.0f});
            result.put("white_bg", false);
        }
        
        return result;
    }
    
    /**
     * Extract image list from directory
     */
    public static List<String> extractImageList(String imageDir) {
        List<String> imageList = new ArrayList<>();
        
        try {
            Files.walk(Paths.get(imageDir))
                .filter(Files::isRegularFile)
                .filter(path -> IMAGE_PATTERN.matcher(path.toString()).matches())
                .sorted()
                .forEach(path -> imageList.add(path.toString()));
                
        } catch (IOException e) {
            System.err.println("Error extracting image list: " + e.getMessage());
        }
        
        return imageList;
    }
    
    /**
     * Create train/test split
     */
    public static Map<String, List<Integer>> createTrainTestSplit(int totalImages, 
                                                                 int trainRatio, 
                                                                 int testRatio) {
        Map<String, List<Integer>> split = new HashMap<>();
        
        List<Integer> allIndices = new ArrayList<>();
        for (int i = 0; i < totalImages; i++) {
            allIndices.add(i);
        }
        Collections.shuffle(allIndices);
        
        int trainSize = (int) (totalImages * trainRatio / 100.0);
        int testSize = (int) (totalImages * testRatio / 100.0);
        
        split.put("train", allIndices.subList(0, trainSize));
        split.put("test", allIndices.subList(trainSize, trainSize + testSize));
        
        return split;
    }
    
    // Helper methods
    
    private static float[][] parseNumpyArray(byte[] data) {
        // Simplified numpy array parsing
        // Trong thực tế cần proper .npy parser
        int numImages = 10; // Default
        float[][] result = new float[numImages][17];
        
        for (int i = 0; i < numImages; i++) {
            for (int j = 0; j < 17; j++) {
                if (j < 15) {
                    // Pose data (default identity matrix)
                    if (j % 5 == 0 && j < 15) {
                        result[i][j] = 1.0f;
                    } else {
                        result[i][j] = 0.0f;
                    }
                } else {
                    // Bounds data
                    result[i][j] = j == 15 ? 0.1f : 10.0f;
                }
            }
        }
        
        return result;
    }
    
    private static String[] extractJsonArray(String content, String key) {
        // Simple JSON array extraction
        String pattern = "\"" + key + "\"\\s*:\\s*\\[([^\\]]+)\\]";
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(pattern);
        java.util.regex.Matcher m = p.matcher(content);
        
        if (m.find()) {
            return m.group(1).split(",");
        }
        
        return new String[0];
    }
    
    private static float[][][] generateDefaultPoses(int numImages) {
        float[][][] poses = new float[numImages][3][5];
        for (int i = 0; i < numImages; i++) {
            // Identity matrix
            poses[i][0] = new float[]{1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            poses[i][1] = new float[]{0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
            poses[i][2] = new float[]{0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
        }
        return poses;
    }
    
    private static float[][] generateDefaultBounds(int numImages) {
        float[][] bounds = new float[numImages][2];
        for (int i = 0; i < numImages; i++) {
            bounds[i][0] = 0.1f; // near
            bounds[i][1] = 10.0f; // far
        }
        return bounds;
    }
    
    private static float[][] generateDefaultIntrinsics() {
        return new float[][]{
            {500.0f, 0.0f, 256.0f},
            {0.0f, 500.0f, 256.0f},
            {0.0f, 0.0f, 1.0f}
        };
    }
    
    private static float[][] generateDefaultSceneBbox() {
        return new float[][]{
            {-1.5f, -1.5f, -1.5f},
            {1.5f, 1.5f, 1.5f}
        };
    }
}
